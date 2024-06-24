# FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
# FROM ubuntu:22.04
# # FROM python:3.8.19
# RUN apt-get update
# RUN apt-get install python3.8
# RUN apt-get install python3-pip=23.3.1
# RUN CUDA_VISIBLE_DEVICES=0 python main.py --dataset KonIQ-10k --resize --exp_id 0 -lr 1e-6 -bs 4 -e 5 --ft_lr_ratio 0.1 -arch resnet50 --loss_type norm-in-norm --p 1 --q 2 --activation relu -prune 0.1 -t_prune pls --pbar --debug
# RUN pip --version
# RUN pip install -r requirements.txt
#docker run --rm --name t0 -v $(pwd)/:/workspace -it test:v13

# FROM continuumio/anaconda3


# ENV NAME=smooth

# RUN conda create -n NAME python=3.8 pip=23.3 -y


# FROM nvidia/cuda:12.2.2-base-ubuntu22.04
FROM nvidia/cuda:11.6.1-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home

# RUN [ -d checkpoints/ ] || echo "NO checkpoints/ dir"; exit 1;
# RUN [ -d KonIQ-10k/ ] || echo "NO KonIQ-10k/ dir"; exit 1;
# RUN [ -d weights/ ] || echo "NO weights/ dir"; exit 1;
# Install system dependencies
RUN apt-get update
# RUN apt install software-properties-common -y
# RUN apt-get update && add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y \ 
        python3.8 \
        python3-pip


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html


# RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update
# RUN apt-get install -y \
#         # git \
#         python3.8 \
#         python3-pip=23.3.1 
        # python3-dev \
        # python3-opencv \
        # libglib2.0-0
# Install any python packages you need
# COPY requirements.txt requirements.txt

# RUN python3 -m pip install -r requirements.txt

# Upgrade pip
# RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
# RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
# RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Set the working directory
# WORKDIR /app

# Set the entrypoint
# ENTRYPOINT [ "python3" ]



#docker run --name smooth --rm --gpus all -it -v $(pwd):/home -v /media/igrpc/sdb1/KonIQ-10k:/home/KonIQ-10k:ro VG:v01