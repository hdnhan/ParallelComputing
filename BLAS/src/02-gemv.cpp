#include <cblas.h>
#include <omp.h>

#include <cstring>
#include <future>
#include <iostream>

#include "common.h"

#ifdef CUDA_FOUND
#include "02-gemv-kernel.cu"
#endif

typedef double T;  // float vs double

/*
Calculate: y = alpha * A * x + beta * y
*/

// Normal matrix-vector multiplication
void gemv(T const alpha, T const *hA, T const *hx, T const beta, T *hy, uint64_t const rows,
          uint64_t const cols) {
    for (uint64_t r = 0; r < rows; ++r) {
        T tmp = 0.0;
        for (uint64_t c = 0; c < cols; ++c) tmp += hA[r * cols + c] * hx[c];
        hy[r] = alpha * tmp + beta * hy[r];
    }
}

// OpenMP matrix-vector multiplication
void openmp_gemv(T const alpha, T const *hA, T const *hx, T const beta, T *hy, uint64_t const rows,
                 uint64_t const cols) {
#pragma omp parallel for schedule(static)
    for (uint64_t r = 0; r < rows; ++r) {
        T tmp = 0.0;
        for (uint64_t c = 0; c < cols; ++c) tmp += hA[r * cols + c] * hx[c];
        hy[r] = alpha * tmp + beta * hy[r];
    }
}

// OpenBLAS matrix-vector multiplication
void openblas_gemv(T const alpha, T const *hA, T const *hx, T const beta, T *hy, uint64_t const rows,
                   uint64_t const cols) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, alpha, hA, cols, hx, 1, beta, hy, 1);
}

int main() {
    common::Timer t("GEMV");
    std::cout << "\n\nGEMV: y = alpha * A * x + beta * y"
              << "\n";

    uint64_t const rows = 1 << 14;
    uint64_t const cols = 1 << 12;
    T const alpha = 2.0;
    T const beta = 1.0;

    T *hA = (T *)malloc(rows * cols * sizeof(T));
    T *hx = (T *)malloc(cols * sizeof(T));
    T *hy = (T *)malloc(rows * sizeof(T));
    T *hr1 = (T *)malloc(rows * sizeof(T));
    T *hr2 = (T *)malloc(rows * sizeof(T));
    T *hr3 = (T *)malloc(rows * sizeof(T));
#ifdef CUDA_FOUND
    T *hr4 = (T *)malloc(rows * sizeof(T));
    T *hr5 = (T *)malloc(rows * sizeof(T));
#endif

    // export OMP_NUM_THREADS=8
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";
    {
        common::Timer t("Init    ");
        common::init(hA, rows, cols, 0., 1.);
        common::init(hx, cols, 0., 1.);
        common::init(hy, rows, 0., 1.);
        memcpy(hr1, hy, rows * sizeof(T));
        memcpy(hr2, hy, rows * sizeof(T));
        memcpy(hr3, hy, rows * sizeof(T));
#ifdef CUDA_FOUND
        memcpy(hr4, hy, rows * sizeof(T));
        memcpy(hr5, hy, rows * sizeof(T));
#endif
    }

    {
        common::Timer t("Normal  ");
        gemv(alpha, hA, hx, beta, hr1, rows, cols);
    }

    {
        common::Timer t("OpenMP  ");
        openmp_gemv(alpha, hA, hx, beta, hr2, rows, cols);
    }

    {
        common::Timer t("OpenBLAS");
        openblas_gemv(alpha, hA, hx, beta, hr3, rows, cols);
    }

#ifdef CUDA_FOUND
    {
        common::Timer timer("CUDA    ");
        cuda_gemv(alpha, hA, hx, beta, hr4, rows, cols);
    }

    {
        common::Timer timer("cuBLAS  ");
        cublas_gemv(alpha, hA, hx, beta, hr5, rows, cols);
    }
#endif

    std::cout << "\n";
    std::cout << "diff(hr1, hr2): " << common::diff(hr1, hr2, rows) << "\n";
    std::cout << "diff(hr1, hr3): " << common::diff(hr1, hr3, rows) << "\n";
#ifdef CUDA_FOUND
    std::cout << "diff(hr1, hr4): " << common::diff(hr1, hr4, rows) << "\n";
    std::cout << "diff(hr1, hr5): " << common::diff(hr1, hr5, rows) << "\n";
#endif

    free(hA);
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