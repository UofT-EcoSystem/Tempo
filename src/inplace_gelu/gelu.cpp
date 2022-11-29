#include <torch/extension.h>
#include <iostream>
#include <vector>

#define MIN -0.751791631228899

std::vector<torch::Tensor> gelu_cuda_forward(
    torch::Tensor input);

std::vector<torch::Tensor> gelu_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor output,
    torch::Tensor mask);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector <torch::Tensor> gelu_forward (torch::Tensor gelu_inp) {
	CHECK_INPUT(gelu_inp);
	// Return GELU and mask of before and after minimum (Allows invertibility)
	return gelu_cuda_forward(gelu_inp);
}

std::vector <torch::Tensor> gelu_backward (torch::Tensor grad_out, torch::Tensor gelu_out, torch::Tensor mask) {
	CHECK_INPUT(grad_out);
	CHECK_INPUT(gelu_out);
	CHECK_INPUT(mask);
	return gelu_cuda_backward(grad_out, gelu_out, mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gelu_forward, "In-place GELU forward");
  m.def("backward", &gelu_backward, "In-place GELU backward");
}
