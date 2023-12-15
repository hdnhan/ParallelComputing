#include <cublas_v2.h>     // cublasHandle_t, cublasCreate_v2, cublasDestroy_v2, cublasSaxpy, cublasDaxpy
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, cudaFree

#include <cstdint>    // uint64_t
#include <stdexcept>  // std::runtime_error

template <typename T>
__global__ void cuda_gemm_kernel(T const alpha, T const *dA, T const *dB, T const beta, T *dC,
                                 uint64_t const rows, uint64_t const N,  uint64_t const cols) {
    uint64_t rid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t cid = blockIdx.y * blockDim.y + threadIdx.y;

    if (rid >= rows || cid >= cols) return;
    T tmp = 0;
    for (int k = 0; k < N; k++) tmp += dA[rid * N + k] * dB[k * cols + cid];
    dC[rid * cols + cid] = alpha * tmp + beta * dC[rid * cols + cid];
}

// CUDA matrix-matrix multiplication
template <typename T>
void cuda_gemm(T const alpha, T const *hA, T const *hB, T const beta, T *hC, uint64_t const rows,
               uint64_t const N, uint64_t const cols) {
    // Allocate device memory
    T *dA, *dB, *dC;
    cudaMalloc(&dA, rows * N * sizeof(T));
    cudaMalloc(&dB, N * cols * sizeof(T));
    cudaMalloc(&dC, rows * cols * sizeof(T));

    // Copy data to the device
    cudaMemcpy(dA, hA, rows * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // CUDA kernel
    // Threads per CTA dimension
    uint64_t const RTHREADS = 32;
    uint64_t const CTHREADS = 32;

    // Blocks per grid dimension
    uint64_t const RBLOCKS = (max(rows, N) + RTHREADS - 1) / RTHREADS;
    uint64_t const CBLOCKS = (max(N, cols) + CTHREADS - 1) / CTHREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(RTHREADS, CTHREADS);
    dim3 blocks(RBLOCKS, CBLOCKS);
    cuda_gemm_kernel<<<blocks, threads>>>(alpha, dA, dB, beta, dC, rows, N, cols);
    cudaMemcpy(hC, dC, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// cuBLAS matrix-matrix multiplication
template <typename T>
void cublas_gemm(T const alpha, T const *hA, T const *hB, T const beta, T *hC, uint64_t const rows,
                 uint64_t const N, uint64_t const cols) {
    // Allocate device memory
    double *dA, *dB, *dC;
    cudaMalloc(&dA, rows * N * sizeof(double));
    cudaMalloc(&dB, N * cols * sizeof(double));
    cudaMalloc(&dC, rows * cols * sizeof(double));

    // Copy data to the device
    cudaMemcpy(dA, hA, rows * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, rows * cols * sizeof(double), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS gemm
    // Calculate: C = (alpha * A) * B + (beta * C)
    // (rows x N) * (N x cols) = (rows x cols)
    // See A and B as row-major matrices and outputs a row-major matrix dC
    if constexpr (std::is_same<T, float>::value)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, N, &alpha, dB, cols, dA, N, &beta, dC,
                    cols);
    else if constexpr (std::is_same<T, double>::value)
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, N, &alpha, dB, cols, dA, N, &beta, dC,
                    cols);
    else
        throw std::runtime_error("cublas_gemm: unsupported type");
    cudaMemcpy(hC, dC, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy_v2(handle);
}
