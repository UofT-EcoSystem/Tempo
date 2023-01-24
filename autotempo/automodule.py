import torch
import torch.nn as nn
from torch import Tensor, device, dtype
import sys
import time
import gc

from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict
from collections import OrderedDict
from transformers import MM, GELUActivation

from tempo.inplace_gelu import InplaceGelu
from tempo.inplace_layernorm import InplaceLayerNorm
from tempo.combined import Combined

class AutoModule(nn.Module):
    def __init__(self, model, inp_size):
        super().__init__()
        self.model = model
        AutoModule.convert_layers(model, inp_size)

    @staticmethod
    def convert_helper(module):
        p1, p2 = (None, None), (None, None)
        for name, child in module.named_children():
            # Do not convert layers that are already quantized
            if isinstance(child, (InplaceGelu, InplaceLayerNorm, Combined)):
                continue
            if isinstance(child, (nn.GELU, GELUActivation)):
                setattr(module, name, InplaceGelu())
            elif isinstance(child, nn.LayerNorm):
                setattr(module, name, InplaceLayerNorm(child.normalized_shape))
            elif (isinstance(child, MM) and isinstance(p1[1], nn.Dropout) and isinstance(p2[1], nn.Softmax)):
                setattr(module, p1[0], nn.Identity())
                setattr(module, p2[0], nn.Identity())
                setattr(module, name, Combined(p2[1].dim, p1[1].p))
            else:
                AutoModule.convert_helper(child)
            p2 = p1
            p1 = (name, child)

    @staticmethod
    def convert_layers(module, inp_size):
        batch_size = 1
        batch_inc = 1
        times = []
        batch_sizes = []
        m = module
        m.cuda()
        while True:
            try:
                sizes = [batch_size] + list(inp_size)
                gc.collect()
                torch.cuda.empty_cache()
                inp = torch.randn(sizes, device='cuda')
                start = time.perf_counter_ns()
                out = m(inp)
                out.backward()
                end = time.perf_counter_ns()
                del out
                del inp
                times.append(end - start)
                batch_sizes.append(batch_size)
                batch_size += batch_inc
                batch_inc *= 2
            except:
                if (batch_inc == 1):
                    break
                else:
                    batch_size -= (batch_inc // 2)
                    batch_inc = 1

        print(list(zip(batch_sizes, times)))
        should_convert = False
        if (len(times) <= 2):
            should_convert = True
        else:
            r1 = (times[-1]/times[-2])
            b1 = (batch_sizes[-1]/batch_sizes[-2])
            r2 = (times[-2]/times[-3])
            b2 = (batch_sizes[-2]/batch_sizes[-3])
            c = 0.85
            if (r1 < c*b1 and r2 < c*b2):
                should_convert = True

        if should_convert:
            AutoModule.convert_helper(module)

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
