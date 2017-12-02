FROM nvidia/cuda:8.0-cudnn6-devel

# Install basic requirements
RUN apt-get update && \
    apt-get install -y --no-install-recommends \ 
        git \
        wget \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-tk \
        protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# Install Python requirements
ADD requirements.txt /opt
RUN pip3 install --upgrade pip && \
    pip3 install --requirement /opt/requirements.txt && \
    rm /opt/requirements.txt

# Clone TensorFlow models repository
WORKDIR /opt
RUN git clone https://github.com/tensorflow/models.git && \
    cd models && \
    git checkout 9b51944bfbae5c971ea9a60226e081a578bbfc3b

# Compile protos
WORKDIR /opt/models/research
RUN protoc object_detection/protos/*.proto --python_out=.

# Export Python PATH
ENV PYTHONPATH /opt/models/research:/opt/models/research/slim   

