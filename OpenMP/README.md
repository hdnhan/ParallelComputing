- Docs: [OpenMP Specifications](https://www.openmp.org/specifications/) and [A Hands-on Introduction to OpenM](https://www.openmp.org/wp-content/uploads/Intro_To_OpenMP_Mattson.pdf).

- OpenMP Directive Format:
    - directive-name: parallel, for, sections, single, task
    - clause is optional
```cpp
#pragma omp directive-name [clause[[,] clause] ...]
```

- High level synchronization:
    - critical
    - atomic
    - barrier
    - ordered
- Low level synchronization
    - flush
    - lock

- Compile and run:
```bash
g++ -fopenmp pi.cpp && OMP_NUM_THREADS=4 ./a.out
```

- [OpenMP Trace/Profile](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#openmp-trace)
```bash
OMP_NUM_THREADS=4 nsys profile --trace=openmp,osrt --stats=true ./a.out
```

- OpenMP version
```bash
echo |cpp -fopenmp -dM |grep -i openmp
# Or
echo _OPENMP | gcc -fopenmp -E -x c - | tail -1
```

## Build and Run
```bash
docker build -t cpu .
docker run -it --rm -v $(pwd):/workspace cpu bash run.sh  
```

## Results (aws ec2 g4dn.xlarge)
```
Estimate pi by integral: pi = 4 * int_0^1 dx / (1 + x^2)
Max threads: 4
Normal  : 2.66078s (2660.78ms)
Parallel: 0.69763s (697.63ms)
Parallel: 0.695478s (695.478ms)
For     : 0.79396s (793.96ms)
diff(pi1, pi2): 3.91243e-13
diff(pi1, pi3): 3.91243e-13
diff(pi1, pi4): 1.89626e-13

Fibonacci(40)
Max threads: 4
Normal : 0.186623s (186.623ms)
Array  : 3e-06s (0.003ms)
Task   : 2.55175s (2551.75ms)
Ordered: 0.000189s (0.189ms)
diff(res1, res2): 0
diff(res1, res3): 0
diff(res1, res4): 0

Dot product
Max threads: 4
Normal  : 0.20559s (205.59ms)
Parallel: 0.059573s (59.573ms)
diff(sum1, sum2): 4.46662e-06

2D Convolution
Max threads: 4
Forward: 2.95976s (2959.76ms)
```
