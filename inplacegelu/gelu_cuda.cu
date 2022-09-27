#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <math.h>

#include <vector>

#define SQRT2 1.414213562373095
#define MIN -0.751791631228899

namespace {

template <typename scalar_t>
__device__ __forceinline__ float gelu_backward(float g, scalar_t o, bool m) {
    float rval;
    // Polynomial approximation of different regimes - no closed-form solution (transcendental)
    // Splined Polynomial see splrep and PPoly from scipy
    // Horner's method do polynomial evalutation in O(n)
    // 2 Cases, before and after x-minimum for inverse
    if (m) {
      if (o > ((float) 0.97742819)) { // (0.97742819, inf)
        if (o < ((float) 3.0)) { // (0.97742819, 3.0)
          float o_translated = o + ((float) 0.9774323423471688);
          rval = 
            o_translated * 
            (((float) 1.40963829e-01) + o_translated * 
            (((float) -2.74712808e-01) + o_translated * 
            (((float) 1.30758747e-01) + o_translated * 
            (((float) -3.34020163e-02) + o_translated * 
            (((float) 2.21484679e-02) + o_translated * 
            (((float) -1.73618287e-02) + o_translated * 
            (((float) 1.13268204e-02) + o_translated * 
            (((float) -7.30845601e-03) + o_translated * 
            (((float) 3.40900094e-03) + o_translated * 
            (((float) -9.19837579e-04) + o_translated * 
            (((float) 1.24505621e-04) + o_translated * ((float) -6.11869959e-06) 
            ))))))))))) + ((float) 1.1079279405461824);
        } else { // [3.0, inf)
          rval = 1.0;
        }
      } else { // (-0.16997121, 0.97742819)
        if (o < ((float) 0.10682432)) { // (-0.16997121, 0.10682432)
          float o_translated = o + ((float) 0.1699712074799012);
          rval = 
            o_translated * 
            (((float) 2.03860722e+01) + o_translated * 
            (((float) -1.51033017e+03) + o_translated * 
            (((float) 7.29893892e+04) + o_translated * 
            (((float) -2.08603338e+06) + o_translated * 
            (((float) 3.75106859e+07) + o_translated * 
            (((float) -4.43847941e+08) + o_translated * 
            (((float) 3.54134638e+09) + o_translated * 
            (((float) -1.91638067e+10) + o_translated * 
            (((float) 6.93203478e+10) + o_translated * 
            (((float) -1.60441793e+11) + o_translated * 
            (((float) 2.14787711e+11) + o_translated * ((float) -1.26477496e+11) 
            )))))))))));
        } else { // (0.10682432, 0.97742819)
          float o_translated = o - ((float) 0.10682674544528972);
          rval = 
            o_translated * 
            (((float) 1.19132403) + o_translated * 
            (((float) -1.44264915) + o_translated * 
            (((float) 1.9529612) + o_translated * 
            (((float) -3.84418714) + o_translated * 
            (((float) 8.2541642) + o_translated * 
            (((float) -16.40234298) + o_translated * 
            (((float) 27.17877296) + o_translated * 
            (((float) -34.79939566) + o_translated * 
            (((float) 32.32044636) + o_translated * 
            (((float) -20.25396347) + o_translated * 
            (((float) 7.6092532) + o_translated * ((float) -1.28943268) 
            ))))))))))) + ((float) 0.6468323960187936);
        }
      }
    } else {
        if (o > ((float) -3.0)) {
          float o_translated = o + ((float) 0.1699712074799012);
          rval = 
            o_translated * 
            (((float) -2.44705742e+01) + o_translated * 
            (((float) 3.64509157e+03) + o_translated * 
            (((float) -3.24039623e+05) + o_translated * 
            (((float) 1.72624707e+07) + o_translated * 
            (((float) -5.87053927e+08) + o_translated * 
            (((float) 1.33643456e+10) + o_translated * 
            (((float) -2.09622261e+11) + o_translated * 
            (((float) 2.29518632e+12) + o_translated * 
            (((float) -1.75056809e+13) + o_translated * 
            (((float) 9.11525632e+13) + o_translated * 
            (((float) -3.08903527e+14) + o_translated * 
            (((float) 6.14002866e+14) + o_translated * ((float) -5.43092154e+14) 
            ))))))))))));
        } else {
          rval = 0.0;
        }
    }
    return rval * g;
}

template <typename scalar_t>
__global__ void gelu_cuda_backward_kernel(
    size_t size,
    const float4* __restrict__ grad_output,
    const float4* __restrict__ output,
    const uchar4* __restrict__ mask,
    float4* grad_input) {
    // Calculate grad_input using output and grad_output only. Uses mask for invertibility and polynomial approximations
    #pragma unroll
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < size; 
         i += blockDim.x * gridDim.x){
        float4 go = grad_output[i];
        float4 o = output[i];
        uchar4 m = mask[i];
        float4 gi;
        gi = make_float4(gelu_backward<scalar_t>(go.x, o.x, (bool) m.x),
                         gelu_backward<scalar_t>(go.y, o.y, (bool) m.y),
                         gelu_backward<scalar_t>(go.z, o.z, (bool) m.z),
                         gelu_backward<scalar_t>(go.w, o.w, (bool) m.w));
        grad_input[i] = gi;
  }
}


template <typename scalar_t>
__device__ __forceinline__ float gelu_forward(float x) {
    /* GELU Hendrycks */
    float rval;
    rval = ((float) 0.5) * x * (((float) 1.0) + erf(x/((float) SQRT2)));
    return rval; 
}

template <typename scalar_t>
__global__ void gelu_cuda_forward_kernel(
    size_t size,
    const float4* __restrict__ input,
    float4* output,
    uchar4* mask) {
    #pragma unroll
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < size; 
         i += blockDim.x * gridDim.x){
        float4 inp = input[i];
        float4 out;
        uchar4 m;
        out = make_float4(gelu_forward<scalar_t>(inp.x),
                          gelu_forward<scalar_t>(inp.y),
                          gelu_forward<scalar_t>(inp.z),
		 	  gelu_forward<scalar_t>(inp.w));

	m = make_uchar4((inp.x >= ((float) MIN)) ? 255 : 0,
                        (inp.y >= ((float) MIN)) ? 255 : 0,
                        (inp.z >= ((float) MIN)) ? 255 : 0,
                        (inp.w >= ((float) MIN)) ? 255 : 0);

	output[i] = out;
	mask[i] = m;
  }
}

} // namespace

std::vector<torch::Tensor> gelu_cuda_forward(
    torch::Tensor input) {

  auto output_options =
    torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA)
    .requires_grad(false);

  auto mask_options =
    torch::TensorOptions()
    .dtype(torch::kBool)
    .layout(torch::kStrided)
    .device(torch::kCUDA)
    .requires_grad(false);

  auto output = torch::empty_like(input, output_options);
  auto mask = torch::empty_like(input, mask_options);

  size_t nelement = torch::numel(input);
  size_t thread_size = 64;

  const dim3 threads(thread_size);
  const dim3 blocks(((nelement / 4) + thread_size - 1) / thread_size); // Ceil Trick: https://stackoverflow.com/questions/62032583/division-round-up-in-c

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_cuda_forward", ([&] {
    gelu_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        nelement/4,
        (float4*)input.data_ptr(),
        (float4*)output.data_ptr(),
        (uchar4*)mask.data_ptr());
  }));

  return {output, mask};
}

std::vector<torch::Tensor> gelu_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor output,
    torch::Tensor mask) {
  
  // Create output tensor

  auto grad_input_options =
    torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA)
    .requires_grad(false);

  auto grad_input = torch::empty_like(output, grad_input_options);

  size_t nelement = torch::numel(output);
  size_t thread_size = 64;

  const dim3 threads(thread_size);
  const dim3 blocks(((nelement / 4) + thread_size - 1) / thread_size); // Ceil Trick: https://stackoverflow.com/questions/62032583/division-round-up-in-c

  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "gelu_cuda_backward", ([&] {
    gelu_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        nelement/4,
        (float4*)grad_output.data_ptr(),
        (float4*)output.data_ptr(),
        (uchar4*)mask.data_ptr(),
        (float4*)grad_input.data_ptr());
  }));

  return {grad_input};
}
