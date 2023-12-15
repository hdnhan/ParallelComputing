FROM nvcr.io/nvidia/cuda:11.5.0-devel-ubuntu20.04 as BASE
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y build-essential cmake
RUN apt-get autoclean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
