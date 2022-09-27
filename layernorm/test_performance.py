import torch
from time import perf_counter as T
from layernorm_python import InplaceLayerNorm as InplaceLayerNormPython
from layernorm_cuda import InplaceLayerNorm as InplaceLayerNormCuda

def test_python_version_speed(cfg={}):
    n_iter = cfg.get('iter', 1000)
    device = cfg.get('device', 'cpu')
    ftime1, ftime2, ftime3, btime1, btime2, btime3 = 0, 0, 0, 0, 0, 0
    inplace_layernorm_python = InplaceLayerNormPython(normalized_shape=(756,), eps=1e-5, elementwise_affine=True).to(device)
    inplace_layernorm_cuda = InplaceLayerNormCuda(normalized_shape=(756,), eps=1e-5, elementwise_affine=True).to(device)
    layernorm = torch.nn.LayerNorm(normalized_shape=(756,), eps=1e-5, elementwise_affine=True).to(device)

    input_tensor = torch.rand(32, 128, 756).to(device=device)
    for _ in range(n_iter):
        # python version
        x = input_tensor.detach()
        
        start = T()
        y = inplace_layernorm_python(x)
        ftime1 += T() - start

        z = y.sum()
        start = T()
        z.backward()
        btime1 += T() - start

        # cuda version
        x2 = input_tensor.detach()
        
        start = T()
        y2 = inplace_layernorm_cuda(x2)
        ftime3 += T() - start

        z2 = y2.sum()
        start = T()
        z2.backward()
        btime3 += T() - start

        # original version
        x1 = input_tensor.detach()
        
        start = T()
        y1 = layernorm(x1)
        ftime2 += T() - start

        z1 = y1.sum()
        start = T()
        z1.backward()
        btime2 += T() - start

    print(f"Iterations: {n_iter}")
    print(f"Pytorch Internel version: [forward] {ftime2:.8f} s | [backward] {btime2:.8f} s")
    print(f"Python version (maunal): [forward] {ftime1:.8f} s | [backward] {btime1:.8f} s")
    print(f"Cuda version (maunal): [forward] {ftime3:.8f} s | [backward] {btime3:.8f} s")

if __name__ == '__main__':
    
    test_speed_cfg = {'iter': 10000, 'device': 'cuda'}
    test_python_version_speed(cfg=test_speed_cfg)