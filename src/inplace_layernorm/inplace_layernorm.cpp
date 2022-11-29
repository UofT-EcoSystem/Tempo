#include <ATen/native/layer_norm.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>
#include <iostream>

// CUDA declarations

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
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layernorm_forward_cuda(
    torch::Tensor input,
    c10::IntArrayRef normalized_shape,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    return at::native_layer_norm(input, normalized_shape, weight, bias, eps);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layernorm_backward_cuda(
    const at::Tensor& dY,
    const at::Tensor& output,
    at::IntArrayRef normalized_shape,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight_opt /* optional */,
    const c10::optional<at::Tensor>& bias_opt /* optional */,
    std::array<bool, 3> grad_input_mask
) {
    c10::MaybeOwned<at::Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
    const at::Tensor& weight = *weight_maybe_owned;
    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor& bias = *bias_maybe_owned;

    auto M_N = at::native::_check_layer_norm_inputs(output, normalized_shape, weight, bias);
    auto M = M_N.first;
    auto N = M_N.second;
    auto Y = output.expect_contiguous();
    auto gamma = weight.expect_contiguous();
    auto beta = bias.expect_contiguous();

    at::Tensor dX;
    at::Tensor dgamma;
    at::Tensor dbeta;
    if (grad_input_mask[0]) {
      dX = at::native::empty_like(
          *Y,
          c10::nullopt /* dtype */,
          c10::nullopt /* layout */,
          c10::nullopt /* device */,
          c10::nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (grad_input_mask[1]) {
      dgamma = M > 0 ? at::native::empty_like(
                          *gamma,
                          c10::nullopt /* dtype */,
                          c10::nullopt /* layout */,
                          c10::nullopt /* device */,
                          c10::nullopt /* pin_memory */,
                          LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                    : at::native::zeros_like(
                          *gamma,
                          c10::nullopt /* dtype */,
                          c10::nullopt /* layout */,
                          c10::nullopt /* device */,
                          c10::nullopt /* pin_memory */,
                          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (grad_input_mask[2]) {
      dbeta = M > 0 ? at::native::empty_like(
                          *beta,
                          c10::nullopt /* dtype */,
                          c10::nullopt /* layout */,
                          c10::nullopt /* device */,
                          c10::nullopt /* pin_memory */,
                          LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                    : at::native::zeros_like(
                          *beta,
                          c10::nullopt /* dtype */,
                          c10::nullopt /* layout */,
                          c10::nullopt /* device */,
                          c10::nullopt /* pin_memory */,
                          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (M > 0) {
      layernorm_backward_cuda_kernel_impl(
          dY, *Y, *gamma, *beta, rstd, M, N, &dX, &dgamma, &dbeta);
    }
    return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &layernorm_forward_cuda, "LayerNorm forward (CUDA)");
  m.def("backward", &layernorm_backward_cuda, "LayerNorm backward (CUDA)");
}