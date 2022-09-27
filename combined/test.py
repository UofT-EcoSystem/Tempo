import torch
from combined import CombinedFunction, combinedFunction
#from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd.profiler import profile
import combined_cpp
import time, tqdm

def combinedFunctionBasicTest():
    x = CombinedFunction(-1, 0.1).cuda()
    z = torch.randn(9, 9, 8, 4).cuda()
    y = torch.randn(9, 9, 8, 8).cuda()
    
    print(f'\n Basic Test Shape (9, 9, 8, 4): {x(y, z).shape}\n')

def combinedFunctionForwardTest():
    x = CombinedFunction(-1, 0.0).cuda()
    dropout = torch.nn.Dropout(0.0)
    
    y = torch.randn(2, 2, 2, 2).cuda()
    z = torch.randn(2, 2, 2, 2).cuda()
    
    a = x (y, z)
    
    b = torch.nn.Softmax(dim=-1)(y)
    b = dropout(b)
    b = torch.matmul(b, z)
    
    print(f'\n Forward Is Close: \n {torch.isclose(a, b)} \n ')

def combinedFunctionBackwardTest():
    x = CombinedFunction(-1, 0.1).cuda()
    x.train()
    dropout = torch.nn.Dropout(0.1)
    
    y = torch.randn(32, 12, 128, 128).cuda()
    y.requires_grad=True
    y.retain_grad()
    z = torch.randn(32, 12, 128, 128).cuda()
    z.requires_grad=True
    z.retain_grad()
    
    z2 = z.detach()
    z2.requires_grad=True
    z2.retain_grad()
    y2 = y.detach()
    y2.requires_grad=True
    y2.retain_grad()

    with profile(use_cuda=True) as ourProfFwd:
        a = x(y2, z2)
    a = torch.sum(a)
    with profile(use_cuda=True) as ourProfBwd: 
#    with profile(activities=[
#        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as ourProfBwd:
        start = time.time()
        a.backward()
        delta = time.time() - start
        print (f"Our Backward Time: {delta * 1000} ms")
    with profile(use_cuda=True) as regProfFwd:
        b = torch.nn.Softmax(dim=-1)(y)
        b = dropout(b)
        b = torch.matmul(b, z)
    b = torch.sum(b)
    with profile(use_cuda=True) as regProfBwd:
#    with profile(activities=[
#        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as regProfBwd:
        start = time.time()
        b.backward()
        delta = time.time() - start
        print (f"Pytorch Backward Time: {delta * 1000} ms")
    
    print(ourProfFwd.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print(ourProfBwd.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print(regProfFwd.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print(regProfBwd.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    ourProfBwd.export_chrome_trace("ourtrace.json")
    regProfBwd.export_chrome_trace("regtrace.json")
    print(f'\n Backward z Is Close: \n {torch.isclose(z.grad, z2.grad)} \n\n Backward y Is Close: \n {torch.isclose(y.grad, y2.grad)}\n')

def combinedCPPSpeedTest():
    y = torch.randn(32, 12, 128, 128, requires_grad=True).cuda()
    y.retain_grad()
    z = torch.randn(32, 12, 128, 128, requires_grad=True).cuda()
    z.retain_grad()
    g = torch.randn(32, 12, 128, 128, requires_grad=True).cuda()
    g.retain_grad()

    forward, backward = 0.0, 0.0

    for _ in tqdm.tqdm(range(100000)):
        start = time.time()
        output = combined_cpp.forward(-1, 0.1, y, z)
        forward += time.time() - start

        mult_out, soft_out, mask, mult_inp2 = output
        
        start = time.time()
        combined_cpp.backward(g, -1, 0.1, soft_out, mask, mult_inp2)
        backward += time.time() - start
    
    print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))

if __name__ == "__main__":
    torch.cuda.manual_seed(1)
#    torch.backends.cudnn.deterministic = True
    
    combinedFunctionBasicTest()
    combinedFunctionForwardTest()
    combinedFunctionBackwardTest()
    combinedCPPSpeedTest()
