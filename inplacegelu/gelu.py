import torch
import gelu_cuda

class gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        output, mask = gelu_cuda.forward(inp)
        ctx.save_for_backward(output, mask)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        output, mask = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_inp, = gelu_cuda.backward(grad_out, output, mask)
        return grad_inp

class Gelu(torch.nn.Module):
    def __init__(self):
        super(Gelu, self).__init__()

    def forward(self, inp):
        return gelu.apply(inp)
