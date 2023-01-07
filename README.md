# Tempo

Official implementation of NeurIPS 2022 paper: "[Tempo: Accelerating Transformer-Based Model Training through Memory Footprint Reduction](https://arxiv.org/abs/2210.10246)".

Tempo is an approach to efficiently use accelerator (e.g. GPU) memory resources for training Transformer-based models. It provides drop-in replacements for the GELU, LayerNorm, and Attention layers, reducing the memory usage and ultimately leading to more efficient training.

This repository contains a [PyTorch](https://pytorch.org/) implementation of Tempo, as well as some of the training scripts to reproduce the BERT training/fine-tuning throughput results shown in our paper. 
## Overview

- `src`: Source files of Tempo
- `tempo`: Python interface of Tempo
- `autotempo`: Preliminary effort to enable Tempo to be automatically applied to Transformer models
- `NVIDIA_DeepLearningExample_BERT`: Modified throughput evaluation scripts based on [*DeepLearningExamples/PyTorch/LanguageModeling/BERT/*](https://github.com/NVIDIA/DeepLearningExamples/tree/128ecbe4f8ee0588112a854dabbbd7dc83751e87/PyTorch/LanguageModeling/BERT). The commit we use is 128ecbe4f8ee0588112a854dabbbd7dc83751e87.

**Updates:**

1. We make Tempo compatible with FP16 (Note: the end-to-end throughput is not as good as FP32 due to the unupdated polynomial fitting strategy for InplaceGelu-FP16).
2. We update our code with new version of *DeepLearningExamples/PyTorch/LanguageModeling/BERT*.

## Quick Usage of Tempo

Clone this repo and run installation
```bash
git clone https://github.com/UofT-EcoSystem/Tempo.git
cd Tempo
python setup.py install
```

You can easily import our modules to replace original modules. Take LayerNorm as an example,
```python
from tempo.inplace_gelu import InplaceGelu
from tempo.inplace_layernorm import InplaceLayerNorm
from tempo.combined import Combined

# Original LayerNorm in Transformer model
# self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
# Replace it with Tempo's InplaceLayerNorm
self.LayerNorm = InplaceLayerNorm(config.hidden_size, eps=1e-12)
```

See the example code to apply Tempo (replace modules) for a BERT model at [NVIDIA_DeepLearningExample_BERT/modeling.diff](https://github.com/UofT-EcoSystem/Tempo/blob/main/NVIDIA_DeepLearningExample_BERT/modeling.diff).

## Throughput Benchmarking using Docker

### Docker related

To build docker:
```bash
sudo docker build -t tempo -f Dockerfile .
```

To run a container of tempo:
```bash
sudo docker run --gpus all -it --rm --ipc=host --shm-size=1g --ulimit memlock=-1 --name="tempo" -v $(pwd):/Tempo/ tempo
```

### Preparations

Install Tempo. 
```bash
cd /Tempo/
python setup.py install
```

Install fused_lamb(optional, only needed for pretraining):
```bash
cd /Tempo/NVIDIA_DeepLearningExample_BERT
pip install lamb_amp_opt/
```

Create a soft link for NVIDIA_DeepLearningExample_BERT folder:
```bash
mkdir /workspace/
ln -s /Tempo/NVIDIA_DeepLearningExample_BERT /workspace/bert
cd /workspace/bert/
```

Download the datasets:
```bash
# please check this file to decide datasets to download
bash /workspace/bert/data/create_datasets_from_start.sh
```
Notes: By default, pretraining datasets will not be downloaded. If you want to run the pretraining and to know more about how to download the wikipedia dataset, run `download_wikipedia --help`. Downloading, extracting, and preparing the pretraining dataset may take long.

Download the pretrained checkpoints:
```
mkdir checkpoints
cd checkpoints
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_large_pretraining_amp_lamb/versions/20.03.0/files/bert_large_pretrained_amp.pt'
# wget 'https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_base_pretraining_amp_lamb/versions/19.09.0/files/bert_base.pt'
```

### Modifications on Base Scripts

The major scripts used by tempo is in the `NVIDIA_DeepLearningExample_BERT/scripts/tempo`. We also made some modifications on `run_squad.py` and `run_pretraining.py` to do the benchmarking. You can also search for `by Tempo` to find out our marks when making modifications.

To patch/unpatch Tempo (replace `modeling.py` with `modeling_tempo.py`/`modeling_ori.py`), you can run
```bash
# patch Tempo
./scripts/tempo/patch_tempo.sh

# check transformer version type
./scripts/tempo/check_transformer_version.sh

# unpatch Tempo
./scripts/tempo/unpatch_tempo.sh
```

Check the difference of `modeling_ori.py` and `modeling_tempo.py` at [NVIDIA_DeepLearningExample_BERT/modeling.diff](https://github.com/UofT-EcoSystem/Tempo/blob/main/NVIDIA_DeepLearningExample_BERT/modeling.diff).

### End-to-end Throughput Benchmarking of Fine-tuning

To test the end-to-end throughput of Tempo using SQuAD V1, we provide two example scripts (with largest batch size being searched) for NVIDIA RTX2080Ti and V100 GPUs. See `scripts/tempo/tempo_benchmarking_squad_2080ti.sh` or `scripts/tempo/tempo_benchmarking_squad_v100.sh` for more information. Note that you may need to adjust the batch size based on your own hardware and running environments.


```bash
cd /workspace/bert/
bash scripts/tempo/tempo_benchmarking_squad_2080ti.sh
```

The results (training logs) are stored in `/workspace/bert/results/`.

Here we show the throughput results of finetuning Bert-Large-Uncased on the SQuAD dataset (with sequence length 384 and largest batch size). This benchmarking was done on four RTX2080Ti and four V100 GPUs with 200 or 500 steps according to the model running time.


| Hardware | Batch Size | Precision | Throughput |   Tag    | Batch Size Impr | Throughput Impr |
| :------: | :--------: | :-------: | :--------: | :------: | :-------------: | :-------------: |
|  2080Ti  |     4      |   fp32    |   12.29    | Original |        -        |        -        |
|  2080Ti  |     6      |   fp32    |   16.23    |  Tempo   |     50.00%      |     32.08%      |
|  2080Ti  |     8      |   fp16    |   49.36    | Original |        -        |        -        |
|  2080Ti  |     11     |   fp16    |   61.28    |  Tempo   |     37.50%      |     24.14%      |
|   V100   |     8      |   fp32    |   45.38    | Original |        -        |        -        |
|   V100   |     13     |   fp32    |   50.75    |  Tempo   |     62.50%      |     11.84%      |
|   V100   |     16     |   fp16    |   213.81   | Original |        -        |        -        |
|   V100   |     22     |   fp16    |   203.36   |  Tempo   |     37.50%      |     -4.89%      |

### End-to-end Throughput Benchmarking of Pretraining

To test the throughput of Tempo using the wikipedia dataset, we provide an example script. You can change the hyper-parameters in `scripts/tempo/tempo_benchmarking_pretraining.sh` and run

```bash
cd /workspace/bert/
bash scripts/tempo/tempo_benchmarking_pretraining.sh
```

If you run it for the first time, it takes long to download and preprocess the dataset.

## Citation

If you use Tempo in your work, please cite our NeurIPS'22 publication using the following BibTeX:

```BibTeX
@inproceedings{NeurIPS2022_Tempo,
  title={Tempo: Accelerating Transformer-Based Model Training through Memory Footprint Reduction},
  author={Andoorveedu, Muralidhar and Zhu, Zhanda and Zheng, Bojian and Pekhimenko, Gennady},
  booktitle={Advances in Neural Information Processing Systems},
  year = {2022}
}
```

## Authors

Authors of Tempo: Muralidhar Andoorveedu, Zhanda Zhu, Bojian Zheng, Gennady Pekhimenko.

Tempo is one of the research projects from the [EcoSystem](https://www.cs.toronto.edu/ecosystem/)
group at the [University of Toronto](https://www.utoronto.ca/), [Department of
Computer Science](https://web.cs.toronto.edu/).
