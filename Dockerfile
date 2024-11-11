# Part of the implementation of this container is based on the Amazon SageMaker Apache MXNet container.
# https://github.com/aws/sagemaker-mxnet-container

FROM nvidia/cuda:11.5.2-base-ubuntu20.04

LABEL maintainer="NASA IMPACT"

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.11-dev

RUN apt-get update && apt-get install -y libgl1 python3-pip git libgdal-dev --fix-missing
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /

RUN pip3 install --upgrade pip

# RUN pip3 install GDAL

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

ENV CUDA_VISIBLE_DEVICES=0,1,2

ENV CUDA_HOME=/usr/local/cuda

RUN mkdir models

# Copies code under /opt/ml/code where sagemaker-containers expects to find the script to run
COPY code /opt/ml/code

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM /opt/ml/code/train.py
