#!/bin/bash

mkdir -p build && rm -rf build/*
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

./build/bin/pi
./build/bin/fibonacci
./build/bin/dot
./build/bin/conv2d
