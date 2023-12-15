#include <cblas.h>
#include <omp.h>

#include <cstring>
#include <future>
#include <iostream>

#include "common.h"

#ifdef CUDA_FOUND
#include "03-gemm-kernel.cu"
#endif

typedef double T;  // float vs double

/*
Calculate: C = (alpha * A) * B + (beta * C)
(rows x N) * (N x cols) = (rows x cols)
*/

// Normal matrix-matrix multiplication
template <typename T>
void gemm(T const alpha, T const *hA, T const *hB, T const beta, T *hC, uint64_t const rows, uint64_t const N,
          uint64_t const cols) {
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t c = 0; c < cols; ++c) {
            T tmp = 0.0;
            for (uint64_t i = 0; i < N; ++i) tmp += hA[r * N + i] * hB[i * cols + c];
            hC[r * cols + c] = alpha * tmp + beta * hC[r * cols + c];
        }
    }
}

// OpenMP matrix-matrix multiplication
template <typename T>
void openmp_gemm(T const alpha, T const *hA, T const *hB, T const beta, T *hC, uint64_t const rows,
                 uint64_t const N, uint64_t const cols) {
#pragma omp parallel for schedule(static) collapse(2)
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t c = 0; c < cols; ++c) {
            T tmp = 0.0;
            for (uint64_t i = 0; i < N; ++i) tmp += hA[r * N + i] * hB[i * cols + c];
            hC[r * cols + c] = alpha * tmp + beta * hC[r * cols + c];
        }
    }
}

// OpenBLAS matrix-matrix multiplication
template <typename T>
void openblas_gemm(T const alpha, T const *hA, T const *hB, T const beta, T *hC, uint64_t const rows,
                   uint64_t const N, uint64_t const cols) {
    if constexpr (std::is_same<T, float>::value)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, N, alpha, hA, N, hB, cols, beta,
                    hC, cols);
    else if constexpr (std::is_same<T, double>::value)

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, N, alpha, hA, N, hB, cols, beta,
                    hC, cols);
    else
        throw std::runtime_error("cublas_gemm: unsupported type");
}

int main() {
    common::Timer t("GEMM");
    std::cout << "\n\nGEMM: hC = (alpha * hA) * hB + (beta * hC)"
              << "\n";

    uint64_t const rows = 1 << 10;
    uint64_t const N = 1 << 11;
    uint64_t const cols = 1 << 12;
    T const alpha = 2.0;
    T const beta = 1.0;

    T *hA = (T *)malloc(rows * N * sizeof(T));
    T *hB = (T *)malloc(N * cols * sizeof(T));
    T *hC = (T *)malloc(rows * cols * sizeof(T));
    T *hR1 = (T *)malloc(rows * cols * sizeof(T));
    T *hR2 = (T *)malloc(rows * cols * sizeof(T));
    T *hR3 = (T *)malloc(rows * cols * sizeof(T));
#ifdef CUDA_FOUND
    T *hR4 = (T *)malloc(rows * cols * sizeof(T));
    T *hR5 = (T *)malloc(rows * cols * sizeof(T));
#endif

    // export OMP_NUM_THREADS=8
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";
    {
        common::Timer t("Init    ");
        common::init(hA, rows, N, 0., 1.);
        common::init(hB, N, cols, 0., 1.);
        common::init(hC, rows, cols, 0., 1.);
        memcpy(hR1, hC, rows * cols * sizeof(T));
        memcpy(hR2, hC, rows * cols * sizeof(T));
        memcpy(hR3, hC, rows * cols * sizeof(T));
#ifdef CUDA_FOUND
        memcpy(hR4, hC, rows * cols * sizeof(T));
        memcpy(hR5, hC, rows * cols * sizeof(T));
#endif
    }

    {
        common::Timer t("Normal  ");
        gemm(alpha, hA, hB, beta, hR1, rows, N, cols);
    }

    {
        common::Timer t("OpenMP  ");
        openmp_gemm(alpha, hA, hB, beta, hR2, rows, N, cols);
    }

    {
        common::Timer t("OpenBLAS");
        openblas_gemm(alpha, hA, hB, beta, hR3, rows, N, cols);
    }
#ifdef CUDA_FOUND
    {
        common::Timer timer("CUDA    ");
        cuda_gemm(alpha, hA, hB, beta, hR4, rows, N, cols);
    }
    {
        common::Timer timer("cuBLAS  ");
        cublas_gemm(alpha, hA, hB, beta, hR5, rows, N, cols);
    }
#endif

    std::cout << "\n";
    std::cout << "diff(hr1, hr2): " << common::diff(hR1, hR2, rows, cols) << "\n";
    std::cout << "diff(hr1, hr3): " << common::diff(hR1, hR3, rows, cols) << "\n";
#ifdef CUDA_FOUND
    std::cout << "diff(hr1, hr4): " << common::diff(hR1, hR4, rows, cols) << "\n";
    std::cout << "diff(hr1, hr5): " << common::diff(hR1, hR5, rows, cols) << "\n";
#endif

    free(hA);
    free(hB);
    free(hC);
    free(hR1);
    free(hR2);
    free(hR3);
#ifdef CUDA_FOUND
    free(hR4);
    free(hR5);
#endif

    return 0;
}