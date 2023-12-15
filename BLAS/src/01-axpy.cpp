#include <cblas.h>  // OpenBLAS
#include <omp.h>    // OpenMP

#include <cstring>   // memcpy
#include <iostream>  // std::cout

#include "common.h"

#ifdef CUDA_FOUND
#include "01-axpy-kernel.cu"
#endif

/*
Calculate: y = alpha * x +  y
*/

typedef double T;  // float vs double

// Normal vector-vector addition
template <typename T>
void axpy(T const alpha, T const *x, T *y, uint64_t const N) {
    for (uint64_t i = 0; i < N; ++i) y[i] += alpha * x[i];
}

// OpenMP vector-vector addition
template <typename T>
void openmp_axpy(T const alpha, T const *x, T *y, uint64_t const N) {
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < N; ++i) y[i] += alpha * x[i];
}

// OpenBLAS vector-vector addition
template <typename T>
void openblas_axpy(T const alpha, T const *x, T *y, uint64_t const N) {
    if constexpr (std::is_same<T, float>::value)
        cblas_saxpy(N, alpha, x, 1, y, 1);
    else if constexpr (std::is_same<T, double>::value)
        cblas_daxpy(N, alpha, x, 1, y, 1);
    else
        throw std::runtime_error("Unsupported type");
}

int main() {
    common::Timer timer("AXPY");
    std::cout << "\n\nAXPY: y = alpha * x + y"
              << "\n";

    uint64_t const N = 1 << 26;
    T const alpha = 2.0;

    T *hx = (T *)malloc(N * sizeof(T));
    T *hy = (T *)malloc(N * sizeof(T));
    T *hr1 = (T *)malloc(N * sizeof(T));
    T *hr2 = (T *)malloc(N * sizeof(T));
    T *hr3 = (T *)malloc(N * sizeof(T));
#ifdef CUDA_FOUND
    T *hr4 = (T *)malloc(N * sizeof(T));
    T *hr5 = (T *)malloc(N * sizeof(T));
#endif

    // export OMP_NUM_THREADS=8
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";
    {
        common::Timer t("Init    ");
        common::init(hx, N, (T)0., (T)1.);
        common::init(hy, N, (T)0., (T)1.);
        memcpy(hr1, hy, N * sizeof(T));
        memcpy(hr2, hy, N * sizeof(T));
        memcpy(hr3, hy, N * sizeof(T));
#ifdef CUDA_FOUND
        memcpy(hr4, hy, N * sizeof(T));
        memcpy(hr5, hy, N * sizeof(T));
#endif
    }

    {
        common::Timer timer("Normal  ");
        axpy(alpha, hx, hr1, N);
    }

    {
        common::Timer timer("OpenMP  ");
        openmp_axpy(alpha, hx, hr2, N);
    }

    {
        common::Timer timer("OpenBLAS");
        openblas_axpy(alpha, hx, hr3, N);
    }
#ifdef CUDA_FOUND
    {
        common::Timer timer("CUDA    ");
        cuda_axpy(alpha, hx, hr4, N);
    }

    {
        common::Timer timer("cuBLAS  ");
        cublas_axpy(alpha, hx, hr5, N);
    }
#endif

    std::cout << "\n";
    std::cout << "diff(hr1, hr2): " << common::diff(hr1, hr2, N) << "\n";
    std::cout << "diff(hr1, hr3): " << common::diff(hr1, hr3, N) << "\n";
#ifdef CUDA_FOUND
    std::cout << "diff(hr1, hr4): " << common::diff(hr1, hr4, N) << "\n";
    std::cout << "diff(hr1, hr5): " << common::diff(hr1, hr5, N) << "\n";
#endif

    free(hx);
    free(hy);
    free(hr1);
    free(hr2);
    free(hr3);
#ifdef CUDA_FOUND
    free(hr4);
    free(hr5);
#endif

    return 0;
}