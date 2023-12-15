#!/bin/bash

mkdir -p build && rm -rf build/*
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

./build/bin/01-axpy
./build/bin/01-dot
./build/bin/02-gemv
./build/bin/03-gemm
