# AutoTempo

## Overview

This is very preliminary work intended on making Tempo easily appliable to existing Transformer models.
As such, this work is not fully tested and does not come with any ease-of-use or performance guarantees. 
In addition, the code is not fully commented - this is provided in a purely as-is manner for informational
purposes.

AutoTempo contains two main methods:

### Fast (`automodule.py`)
This method measures the throughput vs batch size curve for an existing model without Tempo applied.
Based on whether the curve appears to be saturated Tempo is applied.

### Smart (`smartmodule.py`)
This method measures the performance for partially applied Tempo. It does this by comparing fully applied Tempo, 
partially applied Tempo, and the base model without Tempo, then doing a sort of "binary search" for the best configuration
in between these.

These methods are applied on a PyTorch modules, adapting the QMOdule code from [ActNN](https://github.com/ucbrise/actnn)
(Chen et al., ICML 2021) for this purpose. The `*convert.py` files contain examples
of how to use these methods. Keep in mind the transformers library must be patched in order for this to work - the matrix
multiply and softmax layers in the Attention section of BERT for example, must be modified to be PyTorch modules in order
for the `combined` layer to be applied properly. We provide a patch that can be applied to 
[`modeling_bert.py`](https://github.com/huggingface/transformers/blob/bd9d51263a92a5d109811af0d72469f6f8f74aba/src/transformers/models/bert/modeling_bert.py)
from the transformers library - the version the patch is applied on is Transformers `v4.10.0`. In addition modifications
must be made to the relevant `__init__.py` files to expose the `MM` layer to AutoTempo.
