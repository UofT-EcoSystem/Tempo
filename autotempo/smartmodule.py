import torch
import torch.nn as nn
from torch import Tensor, device, dtype
import sys
import time
import gc
from copy import deepcopy

from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict
from collections import OrderedDict
from transformers import MM, GELUActivation

from tempo.inplace_gelu import InplaceGelu
from tempo.inplace_layernorm import InplaceLayerNorm
from tempo.combined import Combined

class SmartModule(nn.Module):
    def __init__(self, model, inp_size, d):
        super().__init__()
        self.model = model
        SmartModule.convert(model, inp_size, d)

    @staticmethod
    def convert_helper(module, gelu_count, gcount, layernorm_count, lcount, combined_count, ccount):
        p1, p2 = (None, None), (None, None)
        for name, child in module.named_children():
            # Do not convert layers that are already quantized
            if isinstance(child, (InplaceGelu, InplaceLayerNorm, Combined)):
                continue
            if (isinstance(child, (nn.GELU, GELUActivation)) and gcount < gelu_count):
                setattr(module, name, InplaceGelu())
                gcount += 1
            elif (isinstance(child, nn.LayerNorm) and lcount < layernorm_count):
                setattr(module, name, InplaceLayerNorm(child.normalized_shape))
                lcount += 1
            elif (isinstance(child, MM) and isinstance(p1[1], nn.Dropout) and isinstance(p2[1], nn.Softmax) and ccount < combined_count):
                setattr(module, p1[0], nn.Identity())
                setattr(module, p2[0], nn.Identity())
                setattr(module, name, Combined(p2[1].dim, p1[1].p))
                ccount += 1
            else:
                gcount, lcount, ccount = SmartModule.convert_helper(child, gelu_count, gcount, layernorm_count, lcount, combined_count, ccount)
            p2 = p1
            p1 = (name, child)
        return gcount, lcount, ccount

    @staticmethod
    def convert_layers(module, gelu_count, layernorm_count, combined_count):
        SmartModule.convert_helper(module, gelu_count, 0, layernorm_count, 0, combined_count, 0)

    @staticmethod
    def counter(module, layer):
        count = 0
        for name, child in module.named_children():
            if isinstance(child, layer):
                count += 1
            else:
                count += SmartModule.counter(child, layer)
        return count

    @staticmethod
    def convert(module, inp_size, d):
        def max_batch_times(m):
            print(torch.cuda.memory_allocated())
            batch_size = 1
            batch_inc = 1
            times = []
            batch_sizes = []
            unviable_sizes = []
            while True:
                try:
                    if ((batch_size in unviable_sizes) and batch_inc == 2):
                        break
                    sizes = [batch_size] + list(inp_size)
                    gc.collect()
                    torch.cuda.empty_cache()
                    inp = torch.randint(100, sizes, dtype = d, device='cuda')
                    start = time.perf_counter_ns()
                    out = m(inp)
                    out.pooler_output.sum().backward()
                    end = time.perf_counter_ns()
                    del out
                    del inp
                    times.append(end - start)
                    batch_sizes.append(batch_size)
                    batch_size += batch_inc
                    batch_inc *= 2
                except Exception as e:
                    unviable_sizes.append(batch_size)
                    if (batch_inc == 1):
                        break
                    else:
                        batch_size -= (batch_inc // 2)
                        batch_inc = 1
            return batch_sizes[-1], times[-1]

        gcount = SmartModule.counter(module, (nn.GELU, GELUActivation))
        lcount = SmartModule.counter(module, nn.LayerNorm)
        ccount = SmartModule.counter(module, nn.Softmax)

        low, high = [0, 0, 0], [gcount, lcount, ccount]

        def mid(l, h):
            return ((l[0] + h[0])//2, (l[1] + h[1])//2, (l[2] + h[2])//2)

        def get_time(g, l, c):
            m = deepcopy(module)
            SmartModule.convert_layers(m, g, l, c)
            m.cuda()
            _, t = max_batch_times(m)
            del m
            gc.collect()
            torch.cuda.empty_cache()
            return t
        
        for i in range(3):
            print (f"{i}:{1}")
            ht = get_time(*high)
            print (f"{i}:{2}")
            lt = get_time(*low)
            print (f"{i}:{3}")
            print(mid(low, high))
            mt = get_time(*mid(low, high))
            if (ht > lt):
                high = mid(low, high)
            else:
                low = mid(low, high)

        if (lt > ht):
            final = high
        else:
            final = low

        SmartModule.convert_layers(module, *final)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):
        return super().train()

    def eval(self):
        return super().eval()

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                        strict: bool = True):
        # remove the prefix "model." added by this wrapper
        new_state_dict = OrderedDict([("model." + k,  v) for k, v in state_dict.items()])
        return super().load_state_dict(new_state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super().state_dict(destination, prefix, keep_vars)

        # remove the prefix "model." added by this wrapper
        ret = OrderedDict([(k[6:], v) for k, v in ret.items()])
        return ret
