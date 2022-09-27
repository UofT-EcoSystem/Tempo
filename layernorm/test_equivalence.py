import torch
from layernorm_python import InplaceLayerNorm as Layernorm_python
from layernorm_cuda import InplaceLayerNorm as Layernorm_cuda

# parameters pre-define
torch.manual_seed(1)
device = 'cuda:0'

# manual implement (python version)
x = torch.rand(2, 2, 3) + 0.5
x = x.to(device)
x.requires_grad_()
x.retain_grad()
layernorm_fn_python = Layernorm_python(normalized_shape=x.shape[-1]).to(device)
y = layernorm_fn_python(x)
z = y.sum() * 2
z.backward()

# manual implement (cuda version)
x1 = x.detach()
x1.requires_grad_()
x1.retain_grad()
layernorm_fn_cuda = Layernorm_cuda(normalized_shape=x1.shape[-1]).to(device)
y1 = layernorm_fn_cuda(x1)
z1 = y1.sum() * 2
z1.backward()

# torch function
x2 = x.detach()
x2.requires_grad_()
x2.retain_grad()
layernorm_fn = torch.nn.LayerNorm(normalized_shape=x2.shape[-1]).to(device)
y2 = layernorm_fn(x2)
z2 = y2.sum() * 2
z2.backward()

# T1: test forward equivalence
print("")
print("Test forward pass")
print("y (python, partial):", y[0, 0, 0:5])
print("y1 (cuda, partial):", y1[0, 0, 0:5])
print("y2 (orginal, partial):", y2[0, 0, 0:5])
print(f"Equivalence (python): {torch.allclose(y, y2, atol=1e-5)}")
print(f"Equivalence (cuda): {torch.allclose(y1, y2, atol=1e-5)}")

# T2: test backward equivalence
print("")
print("Test backward pass")
print("dx (python, partial):", x.grad[0, 0, 0:5])
print("dx1 (cuda, partial):", x1.grad[0, 0, 0:5])
print("dx2 (original, partial):", x2.grad[0, 0, 0:5])
print("dweight (python, partial):", layernorm_fn_python.weight.grad[0:5])
print("dweight (cuda, partial):", layernorm_fn_cuda.weight.grad[0:5])
print("dweight (original, partial):", layernorm_fn.weight.grad[0:5])
print("dbias (python, partial):", layernorm_fn_python.bias.grad[0:5])
print("dbias (cuda, partial):", layernorm_fn_cuda.bias.grad[0:5])
print("dbias (original, partial):", layernorm_fn.bias.grad[0:5])
print(f"Equivalence (python): {torch.allclose(x.grad, x2.grad, atol=1e-5)}")
print(f"Equivalence (cuda): {torch.allclose(x1.grad, x2.grad, atol=1e-5)}")