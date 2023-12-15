# Basic Linear Algebra Subprograms (BLAS)

## Build and Run
```bash
# Build the docker image
docker build -t cpu . -f Dockerfile.cpu
docker build -t cuda . -f Dockerfile.cuda

# Run the docker image
docker run -it --rm -v $(pwd):/workspace cpu bash run.sh
docker run -it --rm --gpus all -v $(pwd):/workspace cuda bash run.sh
```

## Results (aws ec2 g4dn.xlarge)
- CPU:
```
AXPY: y = alpha * x + y
Max threads: 4
Init    : 3.88777s (3887.77ms)
Normal  : 0.094279s (94.279ms)
OpenMP  : 0.043446s (43.446ms)
OpenBLAS: 0.041352s (41.352ms)

diff(hr1, hr2): 0
diff(hr1, hr3): 0
AXPY: 4.38485s (4384.85ms)


DOT: r = x * y
Max threads: 4
Init    : 2.78593s (2785.93ms)
Normal  : 0.145696s (145.696ms)
OpenMP  : 0.040412s (40.412ms)
OpenBLAS: 0.038437s (38.437ms)

diff(hr1, hr2): 3.44962e-06
diff(hr1, hr3): 3.5502e-06
DOT: 3.05096s (3050.96ms)


GEMV: y = alpha * A * x + beta * y
Max threads: 4
Init    : 1.51409s (1514.09ms)
Normal  : 0.087223s (87.223ms)
OpenMP  : 0.025055s (25.055ms)
OpenBLAS: 0.019546s (19.546ms)

diff(hr1, hr2): 0
diff(hr1, hr3): 1.3415e-11
GEMV: 1.66752s (1667.52ms)


GEMM: hC = (alpha * hA) * hB + (beta * hC)
Max threads: 4
Init    : 0.400492s (400.492ms)
Normal  : 40.0871s (40087.1ms)
OpenMP  : 35.729s (35729ms)
OpenBLAS: 0.11801s (118.01ms)

diff(hr1, hr2): 0
diff(hr1, hr3): 5.68434e-12
GEMM: 76.3578s (76357.8ms)
```

- CUDA:
```
AXPY: y = alpha * x + y
Max threads: 4
Init    : 3.45731s (3457.31ms)
Normal  : 0.093082s (93.082ms)
OpenMP  : 0.091823s (91.823ms)
OpenBLAS: 0.041516s (41.516ms)
CUDA    : 0.696614s (696.614ms)
cuBLAS  : 0.999589s (999.589ms)

diff(hr1, hr2): 0
diff(hr1, hr3): 0
diff(hr1, hr4): 0
diff(hr1, hr5): 0
AXPY: 5.93539s (5935.4ms)


DOT: r = x * y
Max threads: 4
Init    : 1.5822s (1582.2ms)
Normal  : 0.134681s (134.681ms)
OpenMP  : 0.134894s (134.894ms)
OpenBLAS: 0.038955s (38.955ms)
cuBLAS  : 1.21149s (1211.49ms)

diff(hr1, hr2): 0
diff(hr1, hr3): 2.25566e-06
diff(hr1, hr4): 2.23331e-06
DOT: 3.1413s (3141.3ms)


GEMV: y = alpha * A * x + beta * y
Max threads: 4
Init    : 0.806709s (806.709ms)
Normal  : 0.085698s (85.698ms)
OpenMP  : 0.08621s (86.21ms)
OpenBLAS: 0.019818s (19.818ms)
CUDA    : 0.397742s (397.742ms)
cuBLAS  : 0.765769s (765.769ms)

diff(hr1, hr2): 0
diff(hr1, hr3): 1.40972e-11
diff(hr1, hr4): 9.09495e-13
diff(hr1, hr5): 1.31877e-11
GEMV: 2.18346s (2183.46ms)


GEMM: hC = (alpha * hA) * hB + (beta * hC)
Max threads: 4
Init    : 0.290377s (290.377ms)
Normal  : 37.4066s (37406.6ms)
OpenMP  : 37.4062s (37406.2ms)
OpenBLAS: 0.118921s (118.921ms)
CUDA    : 0.71835s (718.35ms)
cuBLAS  : 0.732619s (732.619ms)

diff(hr1, hr2): 0
diff(hr1, hr3): 6.13909e-12
diff(hr1, hr4): 7.95808e-13
diff(hr1, hr5): 7.95808e-13
GEMM: 76.7124s (76712.4ms)
```