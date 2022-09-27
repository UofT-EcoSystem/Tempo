#include <torch/extension.h>
#include <iostream>
#include <vector>

std::vector <torch::Tensor> combined_forward (int64_t dim, double p, torch::Tensor softmax_inp, torch::Tensor mult_inp2) {
	// Compute the following retaining dropout mask, not retaining mult_inp1 (dropout output)
	// softmax_inp
	//   |
	// Softmax <- dim
	//   |  
	// Dropout <- p
	//   |
	// Multiply -- mult_inp2
	//   |
	auto soft_out = torch::softmax(softmax_inp, dim);
	auto drop_out = torch::_fused_dropout(soft_out, p);
	auto mult_out = torch::matmul(std::get<0>(drop_out), mult_inp2);
	return {mult_out, soft_out, std::get<1>(drop_out), mult_inp2};
}

std::vector <torch::Tensor> combined_backward (torch::Tensor grad_out, int64_t dim, double p, torch::Tensor soft_out, torch::Tensor mask, torch::Tensor mult_inp2) {
	auto mult_factor = 1/p;
	// Partially recompute mult_inp1 and get grad_mult_inp2 (multiply backwards) G_M2 = M1^T .* G_O
	auto grad_mult_inp2 = torch::matmul(torch::_masked_scale(soft_out, mask, mult_factor).transpose(-2, -1), grad_out);
	// Multiply backward G_M1 = M2^T .* G_O
	auto grad_soft_inp = torch::matmul(grad_out, mult_inp2.transpose(-2, -1));
	// Dropout backward
	grad_soft_inp = torch::_masked_scale(grad_soft_inp, mask, mult_factor);
	// Recompute softmax gradient using only output (Huggingface Deberta Style)
	grad_soft_inp = torch::_softmax_backward_data(grad_soft_inp, soft_out, dim, soft_out);
	return {grad_soft_inp, grad_mult_inp2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &combined_forward, "Combined Softmax-Dropout-Multiply forward");
  m.def("backward", &combined_backward, "Combined Softmax-Dropout-Multiply backward");
}
