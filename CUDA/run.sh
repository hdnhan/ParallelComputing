#!/bin/bash

mkdir -p build && rm -rf build/*
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

./build/bin/conv2d