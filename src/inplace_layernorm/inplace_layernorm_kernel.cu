#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <ATen/AccumulateType.h>
#include <ATen/native/cuda/block_reduce.cuh>

#include <iostream>
#include <stdio.h>


namespace at {
namespace native {

namespace {

constexpr int kCUDANumThreads = 256;
constexpr int kColwiseReduceTileSize = 32;

template <typename T>
__global__ void _ComputeHatxCUDAKernel(
    int64_t N,
    const T* Y,
    const T* gamma,
    const T* beta,
    acc_type<T, true>* hatx
) {
    using T_ACC = acc_type<T, true>;
    const int64_t i = blockIdx.x;
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        const int64_t index = i * N + j;
        const T_ACC beta_v = beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[j]);
        const T_ACC gamma_v = gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
        hatx[index] = (static_cast<T_ACC>(Y[index]) - beta_v) / gamma_v;
    }
}

template <typename T>
__global__ void _ComputeInternalGradientsCUDAKernel(
    int64_t N,
    const T* dY,
    const acc_type<T, true>* hatx,
    const T* gamma,
    acc_type<T, true>* ds,
    acc_type<T, true>* db
) {
    using T_ACC = acc_type<T, true>;
    __shared__ T_ACC ds_shared[C10_WARP_SIZE];
    __shared__ T_ACC db_shared[C10_WARP_SIZE];
    const int64_t i = blockIdx.x;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        const int64_t index = i * N + j;
        const T_ACC gamma_v = gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
        sum1 += static_cast<T_ACC>(dY[index]) * hatx[index] * gamma_v;
        sum2 += static_cast<T_ACC>(dY[index]) * gamma_v;
    }
    sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
    sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
    if (threadIdx.x == 0) {
        ds[i] = sum1;
        db[i] = sum2;
    }
}

template <typename T>
__global__ void _LayerNormBackwardCUDAKenrel(
    int64_t N,
    const T* dY,
    const acc_type<T, true>* hatx,
    const T* gamma,
    const T* rstd,
    const acc_type<T, true>* ds,
    const acc_type<T, true>* db,
    T* dX
) {
    using T_ACC = acc_type<T, true>;
    const int64_t i = blockIdx.x;
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        const int64_t index = i * N + j;
        const T_ACC gamma_v = gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
        dX[index] = (static_cast<T_ACC>(dY[index]) * gamma_v - ds[i] * hatx[index] / static_cast<T_ACC>(N) - db[i] / static_cast<T_ACC>(N)) * static_cast<T_ACC>(rstd[i]);
    }
}

template <typename T>
__global__ void _GammaBetaBackwardSimpleCUDAKernel(
    int64_t M,
    int64_t N,
    const T* dY,
    const acc_type<T, true>* hatx,
    T* dg,
    T* db
) {
    using T_ACC = acc_type<T, true>;
    const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        T_ACC sum1 = 0;
        T_ACC sum2 = 0;
        for (int64_t i = 0; i < M; ++i) {
            const int64_t index = i * N + j;
            sum1 += dg == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index]) * hatx[index];
            sum2 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index]);
        }
        if (dg != nullptr) {
            dg[j] = sum1;
        }
        if (db != nullptr) {
            db[j] = sum2;
        }
    }
}

template <typename T>
__global__ void _GammaBetaBackwardCUDAKernel(
    int64_t M,
    int64_t N,
    const T* dY,
    const acc_type<T, true>* hatx,
    T* dg,
    T* db
) {
    using T_ACC = acc_type<T, true>;
    __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
    __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
    const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    T_ACC dg_sum1 = 0;
    T_ACC dg_sum2 = 0;
    T_ACC db_sum1 = 0;
    T_ACC db_sum2 = 0;
    if (j < N) {
        for (int64_t i = threadIdx.y; i < M; i += blockDim.y * 2) {
            const int64_t i1 = i;
            const int64_t i2 = i + blockDim.y;
            const int64_t index1 = i1 * N + j;
            const int64_t index2 = i2 * N + j;
            dg_sum1 += dg == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index1]) * hatx[index1];
            db_sum1 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index1]);
            if (i2 < M) {
                dg_sum2 += dg == nullptr ? T_ACC(0): static_cast<T_ACC>(dY[index2]) * hatx[index2];
                db_sum2 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index2]);
            }
        }
    }
    g_shared[threadIdx.y][threadIdx.x] = dg_sum1;
    g_shared[threadIdx.y + blockDim.y][threadIdx.x] = dg_sum2;
    b_shared[threadIdx.y][threadIdx.x] = db_sum1;
    b_shared[threadIdx.y + blockDim.y][threadIdx.x] = db_sum2;
    __syncthreads();
    T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
    T_ACC sum2 = b_shared[threadIdx.x][threadIdx.y];
    sum1 = cuda_utils::WarpReduceSum(sum1);
    sum2 = cuda_utils::WarpReduceSum(sum2);
    if (threadIdx.x == 0) {
        const int64_t j = blockIdx.x * blockDim.x + threadIdx.y;
        if (j < N) {
            if (dg != nullptr) {
                dg[j] = sum1;
            }
            if (db != nullptr) {
                db[j] = sum2;
            }
        }
    }
    sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
    sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
    sum1 = cuda_utils::WarpReduceSum(sum1);
    sum2 = cuda_utils::WarpReduceSum(sum2);
    if (threadIdx.x == 0) {
        const int64_t j = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
        if (j < N) {
            if (dg != nullptr) {
                dg[j] = sum1;
            }
            if (db != nullptr) {
                db[j] = sum2;
            }
        }
    }
}

template <typename T>
void _LayerNormBackwardKernelImplInternal(
    const torch::Tensor& dY,
    const torch::Tensor& Y,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    const torch::Tensor& rstd,
    int64_t M,
    int64_t N,
    torch::Tensor* dX,
    torch::Tensor* dgamma,
    torch::Tensor* dbeta) {
    using T_ACC = acc_type<T, true>;
    DCHECK_EQ(dY.numel(), M * N);
    DCHECK_EQ(Y.numel(), M * N);
    DCHECK(!gamma.defined() || gamma.numel() == N);
    DCHECK(!beta.defined() || beta.numel() == N);
    DCHECK_EQ(rstd.numel(), M);
    const T* dY_data = dY.template data_ptr<T>();
    const T* Y_data = Y.template data_ptr<T>();
    const T* gamma_data = gamma.defined() ? gamma.template data_ptr<T>() : nullptr;
    const T* beta_data = beta.defined() ? beta.template data_ptr<T>() : nullptr;
    const T* rstd_data = rstd.template data_ptr<T>();
    T* dX_data = dX->defined() ? dX->template data_ptr<T>() : nullptr;
    const auto kAccType =
            (Y.scalar_type() == kHalf || Y.scalar_type() == kBFloat16)
            ? kFloat
            : Y.scalar_type();
    torch::Tensor hatx = at::empty({M, N}, Y.options().dtype(kAccType));
    T_ACC* hatx_data = hatx.template data_ptr<T_ACC>();
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

    if (dX_data != nullptr) {
        torch::Tensor ds = at::empty({M}, Y.options().dtype(kAccType));
        torch::Tensor db = at::empty({M}, Y.options().dtype(kAccType));
        T_ACC* ds_data = ds.template data_ptr<T_ACC>();
        T_ACC* db_data = db.template data_ptr<T_ACC>();

        _ComputeHatxCUDAKernel<T>
            <<<M, cuda_utils::kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
                N, Y_data, gamma_data, beta_data, hatx_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        _ComputeInternalGradientsCUDAKernel<T>
            <<<M, cuda_utils::kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
                N, dY_data, hatx_data, gamma_data, ds_data, db_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        _LayerNormBackwardCUDAKenrel<T><<<M, kCUDANumThreads, 0, cuda_stream>>>(
            N,
            dY_data,
            hatx_data,
            gamma_data,
            rstd_data,
            ds_data,
            db_data,
            dX_data
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (dgamma->defined() || dbeta->defined()) {
        T* dgamma_data = dgamma->defined() ? dgamma->template data_ptr<T>() : nullptr;
        T* dbeta_data = dbeta->defined() ? dbeta->template data_ptr<T>() : nullptr;
        if (M < 512) {
            // For small batch size, do colwise reduce directly.
            const int64_t B = (N + kCUDANumThreads - 1) / kCUDANumThreads;
            _GammaBetaBackwardSimpleCUDAKernel<T>
                <<<B, kCUDANumThreads, 0, cuda_stream>>>(
                    M,
                    N,
                    dY_data,
                    hatx_data,
                    dgamma_data,
                    dbeta_data);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            const int64_t B =
                (N + kColwiseReduceTileSize - 1) / kColwiseReduceTileSize;
            constexpr int kThreadX = kColwiseReduceTileSize;
            constexpr int kThreadY = kColwiseReduceTileSize / 2;
            _GammaBetaBackwardCUDAKernel<T>
                <<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
                    M,
                    N,
                    dY_data,
                    hatx_data,
                    dgamma_data,
                    dbeta_data);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
}

}  // namespace
}  // namespace native
}  // namespace at

void layernorm_backward_cuda_kernel_impl(
    const torch::Tensor& dY,
    const torch::Tensor& Y,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    const torch::Tensor& rstd,
    int64_t M,
    int64_t N,
    torch::Tensor* dX,
    torch::Tensor* dgamma,
    torch::Tensor* dbeta
) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        Y.scalar_type(),
        "_LayerNormBackwardKernelImpl",
        [&]() {
            at::native::_LayerNormBackwardKernelImplInternal<scalar_t>(
                dY, Y, gamma, beta, rstd, M, N, dX, dgamma, dbeta);
        });
}