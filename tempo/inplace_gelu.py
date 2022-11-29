import torch
from tempo.backend import inplace_gelu

class InplaceGeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        output, mask = inplace_gelu.forward(inp)
        ctx.save_for_backward(output, mask)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        output, mask = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_inp, = inplace_gelu.backward(grad_out, output, mask)
        return grad_inp

class InplaceGelu(torch.nn.Module):
    def __init__(self):
        super(InplaceGelu, self).__init__()

    def forward(self, inp):
        return InplaceGeluFunction.apply(inp)
