import torch
import gelu

tensor = torch.randn(2, 2, 2).cuda()
tensor.requires_grad=True
tensor.retain_grad()
    
tensor2 = tensor.detach()
tensor2.requires_grad=True
tensor2.retain_grad()

x = gelu.Gelu()
y = x(tensor)
z = y.sum()
z.backward()

print(tensor.grad)

y = torch.nn.functional.gelu(tensor2)
z = y.sum()
z.backward()

print(tensor2.grad)
