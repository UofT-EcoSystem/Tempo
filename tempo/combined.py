import torch
from tempo.backend import combined

class CombinedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, softmax_inp, soft_dim, p, mult_inp2):
        outputs = combined.forward(soft_dim, p, softmax_inp, mult_inp2)
        mult_out, soft_out, mask, mult_inp2 = outputs
        ctx.save_for_backward(soft_out, mask, mult_inp2)
        ctx.other = (soft_dim, p)
        return mult_out

    @staticmethod
    def backward(ctx, grad_out):
        outputs = combined.backward(grad_out, *ctx.other, *ctx.saved_tensors)
        grad_soft_inp, grad_mult_inp = outputs
        return grad_soft_inp, None, None, grad_mult_inp

class Combined(torch.nn.Module):
    def __init__(self, dim, p):
        super(Combined, self).__init__()
        self.dim, self.p = dim, p

    def forward(self, softmax_inp, mult_inp2):
        return CombinedFunction.apply(softmax_inp, self.dim, 1-self.p, mult_inp2)
