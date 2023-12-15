#include <cblas.h>
#include <omp.h>

#include <cstring>
#include <future>
#include <iostream>

#include "common.h"

#ifdef CUDA_FOUND
#include "01-dot-kernel.cu"
#endif

typedef double T;  // float vs double

/*
Calculate: r = x * y
*/

// Normal vector-vector dot product
template <typename T>
T dot(T const *hx, T const *hy, uint64_t const N) {
    T hr = 0;
    for (uint64_t i = 0; i < N; ++i) hr += hx[i] * hy[i];
    return hr;
}

// OpenMP vector-vector dot product
template <typename T>
T openmp_dot(T const *hx, T const *hy, uint64_t const N) {
    T hr = 0;
#pragma omp parallel for schedule(static) reduction(+ : hr)
    for (uint64_t i = 0; i < N; ++i) hr += hx[i] * hy[i];
    return hr;
}

// OpenBLAS vector-vector dot product
template <typename T>
T openblas_dot(T const *hx, T const *hy, uint64_t const N) {
    if constexpr (std::is_same<T, float>::value)
        return cblas_sdot(N, hx, 1, hy, 1);
    else if constexpr (std::is_same<T, double>::value)
        return cblas_ddot(N, hx, 1, hy, 1);
    else
        throw std::runtime_error("Unsupported type");
}

int main() {
    common::Timer timer("DOT");
    std::cout << "\n\nDOT: r = x * y"
              << "\n";

    uint64_t const N = 1 << 26;

    T *hx = (T *)malloc(N * sizeof(T));
    T *hy = (T *)malloc(N * sizeof(T));
    T hr1 = 0, hr2 = 0, hr3 = 0;
#ifdef CUDA_FOUND
    T hr4 = 0;
#endif

    // export OMP_NUM_THREADS=8
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";
    {
        common::Timer t("Init    ");
        common::init(hx, N, 0., 1.);
        common::init(hy, N, 0., 1.);
    }

    {
        common::Timer t("Normal  ");
        hr1 = dot(hx, hy, N);
    }

    {
        common::Timer t("OpenMP  ");
        hr2 = openmp_dot(hx, hy, N);
    }

    {
        common::Timer t("OpenBLAS");
        hr3 = openblas_dot(hx, hy, N);
    }

#ifdef CUDA_FOUND
    {
        common::Timer timer("cuBLAS  ");
        hr4 = cublas_dot(hx, hy, N);
    }

#endif

    std::cout << "\n";
    std::cout << "diff(hr1, hr2): " << std::abs(hr1 - hr2) << "\n";
    std::cout << "diff(hr1, hr3): " << std::abs(hr1 - hr3) << "\n";
#ifdef CUDA_FOUND
    std::cout << "diff(hr1, hr4): " << std::abs(hr1 - hr4) << "\n";
#endif

    free(hx);
    free(hy);

    return 0;
}