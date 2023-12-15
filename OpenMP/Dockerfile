FROM debian:12.2 as builder
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential cmake 

WORKDIR /workspace
