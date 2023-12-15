#include <cublas_v2.h>     // cublasHandle_t, cublasCreate_v2, cublasDestroy_v2, cublasSaxpy, cublasDaxpy
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, cudaFree

#include <cstdint>    // uint64_t
#include <stdexcept>  // std::runtime_error

template <typename T>
__global__ void cuda_gemv_kernel(T const alpha, T const *dA, T const *dx, T const beta, T *dy,
                                 uint64_t const rows, uint64_t const cols) {
    uint64_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < rows) {
        T tmp = 0;
        for (uint64_t k = 0; k < cols; k++) tmp += dA[tid * cols + k] * dx[k];
        dy[tid] = alpha * tmp + beta * dy[tid];
    }
}

// CUDA matrix-vector multiplication
template <typename T>
void cuda_gemv(T const alpha, T const *hA, T const *hx, T const beta, T *hy, uint64_t const rows,
               uint64_t const cols) {
    // Allocate device memory
    T *dA, *dx, *dy;
    cudaMalloc(&dA, rows * cols * sizeof(T));
    cudaMalloc(&dx, cols * sizeof(T));
    cudaMalloc(&dy, rows * sizeof(T));

    // Copy host memory to device memory
    cudaMemcpy(dA, hA, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, rows * sizeof(T), cudaMemcpyHostToDevice);

    // CUDA kernel
    uint64_t const threads = 256;
    uint64_t const blocks = (rows + threads - 1) / threads;
    cuda_gemv_kernel<<<blocks, threads>>>(alpha, dA, dx, beta, dy, rows, cols);

    // Copy device memory to host memory
    cudaMemcpy(hy, dy, rows * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dA);
    cudaFree(dx);
    cudaFree(dy);
}

// cuBLAS matrix-vector multiplication
template <typename T>
void cublas_gemv(T const alpha, T const *hA, T const *hx, T const beta, T *hy, uint64_t const rows,
                 uint64_t const cols) {
    // Allocate device memory
    T *dA, *dx, *dy;
    cudaMalloc(&dA, rows * cols * sizeof(T));
    cudaMalloc(&dx, cols * sizeof(T));
    cudaMalloc(&dy, rows * sizeof(T));

    // Copy host memory to device memory
    cudaMemcpy(dA, hA, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, rows * sizeof(T), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS gemv
    // Calculate: y = alpha * A * x + beta * y
    if constexpr (std::is_same<T, float>::value)
        cublasSgemv(handle, CUBLAS_OP_T, cols, rows, &alpha, dA, cols, dx, 1, &beta, dy, 1);
    else if constexpr (std::is_same<T, double>::value)
        cublasDgemv(handle, CUBLAS_OP_T, cols, rows, &alpha, dA, cols, dx, 1, &beta, dy, 1);
    else
        throw std::runtime_error("Unsupported type");
    // Copy device memory to host memory
    cudaMemcpy(hy, dy, rows * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dA);
    cudaFree(dx);
    cudaFree(dy);
    cublasDestroy_v2(handle);
}
