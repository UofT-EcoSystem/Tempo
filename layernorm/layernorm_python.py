import torch
import numbers
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class InplaceLayerNormFunction(autograd.Function):
    '''
    Inplace LayerNorm Function
    FIXME: Currently, it works according to the setting of LayerNorm in Bert.
    '''
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps=1e-5):
        mean = torch.mean(input, dim=-1)
        std = torch.var(input, dim=-1, unbiased=False)
        sqrt_std_with_eps = torch.sqrt(std + eps)
        hatx = (input - mean.unsqueeze(-1)) / sqrt_std_with_eps.unsqueeze(-1)
        output = hatx * weight + bias
        ctx.save_for_backward(output, weight, bias, sqrt_std_with_eps)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, weight, bias, sqrt_std_with_eps = ctx.saved_tensors
        weight = weight.expand_as(output)
        m = output.shape[-1]
        hatx = (output - bias) / weight

        # compute dx
        ds = (grad_output * weight * hatx).sum(-1, keepdim=True)
        db = (grad_output * weight).sum(-1, keepdim=True)
        dx = (grad_output * weight - ds * hatx / m - db / m) / sqrt_std_with_eps.unsqueeze(-1)

        # compute dweight and dbias
        dweight = (grad_output * hatx).sum(dim=(0, 1))
        dbias = grad_output.sum(dim=(0, 1))

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