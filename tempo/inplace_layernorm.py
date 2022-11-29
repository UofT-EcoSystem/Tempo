import torch
import numbers
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.parameter import Parameter

from tempo.backend import inplace_layernorm

class InplaceLayerNormFunction(autograd.Function):
    '''
    LayerNorm Function using the output to compute gradient
    '''
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps=1e-5):
        output, _, rstd = inplace_layernorm.forward(input, normalized_shape, weight, bias, eps)
        # FIXME: at::native_layer_norm's output rstd is always float
        if output.dtype == torch.half and rstd.dtype == torch.float:
            rstd = rstd.half()
        ctx.save_for_backward(output, rstd, weight, bias)
        ctx.normalized_shape = normalized_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, rstd, weight, bias = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        grad_mask = [True, True, True]

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        dx, dweight, dbias = inplace_layernorm.backward(grad_output, output, normalized_shape, rstd, weight, bias, grad_mask)

        return dx, None, dweight, dbias, None


class InplaceLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(InplaceLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*self.normalized_shape))
            self.bias = Parameter(torch.Tensor(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return InplaceLayerNormFunction.apply(input, self.normalized_shape, self.weight, self.bias, self.eps)