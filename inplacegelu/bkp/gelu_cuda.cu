#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <crt/math_functions.hpp>

#include <vector>

#define sqrt2 1.41421356237
#define min -0.751791631228899
namespace {
template <typename scalar_t>
__device__ __forceinline__ float gelu(scalar_t x) {
    float rval;
    rval = 0.5*x*(1+(erf(x/sqrt2)));
    return rval;
}

template <typename scalar_t>
__device__ __forceinline__ uint8_t gelu_mask(scalar_t x) {
    float rval;
    rval = x > min ? 255 : 0;
    return rval;
}

template <typename scalar_t>
__device__ __forceinline__ float gelu_backward(float g, scalar_t o, uint8_t m) {
    float rval;
    if (o > 4) {
        rval = 1.0;
    } else {
	if (m == 255) {
	    rval = 0.006268*pow(o, 5) 
		 - 0.09392*pow(o, 4) +
		 + 0.5341*pow(o, 3) +
		 - 1.412*pow(o, 2) +
		 + 1.648*o +
		 + 0.4483;
	} else {
	    rval = 1226.0*pow(o, 4)
		 + 340.8*pow(o, 3)
		 + 39.54*pow(o, 2)
		 + 3.061*o
		 + 3.799e-5;
	}
    }
    return g*rval;
}

template <typename scalar_t>
__global__ void gelu_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<uint8_t,3,torch::RestrictPtrTraits> mask) {
  
  const int batch_size = blockIdx.x;
  const int intermediate = blockIdx.y;
  const int seq_position = threadIdx.x;
  
  output[batch_size][seq_position][intermediate] = gelu<scalar_t>(input[batch_size][seq_position][intermediate]);
  mask[batch_size][seq_position][intermediate] = gelu_mask<scalar_t>(input[batch_size][seq_position][intermediate]);
}

template <typename scalar_t>
__global__ void gelu_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> output,
    const torch::PackedTensorAccessor32<uint8_t,3,torch::RestrictPtrTraits> mask,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_input) {
  const int batch_size = blockIdx.x;
  const int intermediate = blockIdx.y;
  const int seq_position = threadIdx.x;

  grad_input[batch_size][seq_position][intermediate] = gelu_backward<scalar_t>(grad_output[batch_size][seq_position][intermediate], output[batch_size][seq_position][intermediate], mask[batch_size][seq_position][intermediate]);
}

} // namespace

std::vector<torch::Tensor> gelu_cuda(
    torch::Tensor input) {
  
  auto mask_options =
    torch::TensorOptions()
    .dtype(torch::kByte)
    .layout(torch::kStrided)
    .device(torch::kCUDA)
    .requires_grad(false);
  
  auto output_options =
    torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA)
    .requires_grad(false);

  auto mask = torch::empty ({input.size(0),
		                  input.size(1),
				  input.size(2)}, mask_options);

  auto output = torch::empty ({input.size(0),
		                  input.size(1),
				  input.size(2)}, output_options);
  
  const int threads = input.size(1);
  const dim3 blocks(input.size(0), input.size(2));

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_cuda", ([&] {
    gelu_cuda_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
	output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        mask.packed_accessor32<uint8_t,3,torch::RestrictPtrTraits>());
  }));

  return {output, mask};
}

std::vector<torch::Tensor> gelu_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor output,
    torch::Tensor mask) {
  
  auto grad_input_options =
    torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA)
    .requires_grad(false);

  auto grad_input = torch::empty ({output.size(0),
	                           output.size(1),
				   output.size(2)}, grad_input_options);

  const int threads = output.size(1);
  const dim3 blocks(output.size(0), output.size(2));

  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "gelu_cuda_backward", ([&] {
    gelu_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        output.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        mask.packed_accessor32<uint8_t,3,torch::RestrictPtrTraits>(),
        grad_input.packed_accessor32<float,3,torch::RestrictPtrTraits>());
  }));

  return {grad_input};
}
