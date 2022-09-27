FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y vim \
python3.8 \
python3-pip \
python-is-python3
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install transformers==4.8.0 \
datasets==1.11.0 \
accelerate==0.4.0

LABEL version="1.0"
LABEL description="Tempo dockerfile"
