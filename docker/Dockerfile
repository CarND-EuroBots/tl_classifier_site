FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# Install basic requirements
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        python-dev \
        python-pip \
        python-setuptools \
        python-tk \
        protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# Install Python requirements
ADD requirements.txt /opt
RUN pip install --upgrade pip && \
    pip install --requirement /opt/requirements.txt && \
    rm /opt/requirements.txt

# Clone TensorFlow models repository
WORKDIR /opt
RUN git clone https://github.com/tensorflow/models.git && \
    cd models && \
    git checkout edcd29f2dbb4b3eaed387fe17cb5270f867aec42

# Compile protos
WORKDIR /opt/models/research
RUN protoc object_detection/protos/*.proto --python_out=.

# Export Python PATH
ENV PYTHONPATH /opt/models/research:/opt/models/research/slim
