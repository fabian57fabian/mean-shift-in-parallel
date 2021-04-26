FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as base
RUN apt-get update && apt-get install -y nano && apt-get install -y wget

# CPP
WORKDIR /MS_CUDA
RUN apt-get update && apt-get install -y cmake
RUN apt-get install -y libx11-dev
