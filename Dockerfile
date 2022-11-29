FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y vim \
        python3.8 \
        python3-pip \
        python-is-python3
RUN pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install transformers==4.23.0 \
                datasets==2.6.1 \
                accelerate==0.13.1

RUN apt-get update && apt-get install -y git wget unzip \
        pbzip2 pv bzip2 cabextract iputils-ping

## OPENMPI
ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.0
RUN mkdir -p /build && \
    cd /build && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ~ && \
    rm -rf /build

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

RUN echo 'export PATH=/usr/local/mpi/bin:$PATH' >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH' >> /root/.bashrc

# Needs to be in docker PATH if compiling other items & bashrc PATH (later)
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}


## For NVIDIA/DeepLearningExamples
RUN pip install --no-cache-dir \
        tqdm boto3 requests six ipdb h5py wget nltk progressbar onnxruntime tokenizers>=0.7 \
        git+https://github.com/NVIDIA/dllogger \
        git+https://github.com/NVIDIA/lddl.git 
RUN python -m nltk.downloader punkt

# Install apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597

COPY ./NVIDIA_DeepLearningExample_BERT /tmp
RUN pip install -r /tmp/requirements.txt
RUN echo 'export BERT_PREP_WORKING_DIR=/workspace/bert/data' >> /root/.bashrc

RUN mkdir -p /build && \
    cd /build && \
    wget https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2 && \
    tar -jxvf jemalloc-5.3.0.tar.bz2 && \
    cd jemalloc-5.3.0 && \
    ./configure && \
    make && \
    make install && \
    cp -r /usr/local/lib/libjemalloc* /usr/lib/x86_64-linux-gnu/ && \
    rm -rf /build

LABEL version="1.0"
LABEL description="Tempo dockerfile"
