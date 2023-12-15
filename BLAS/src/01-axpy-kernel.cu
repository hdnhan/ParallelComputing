#include <cublas_v2.h>     // cublasHandle_t, cublasCreate_v2, cublasDestroy_v2, cublasSaxpy, cublasDaxpy
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, cudaFree

#include <cstdint>    // uint64_t
#include <stdexcept>  // std::runtime_error

template <typename T>
__global__ void cuda_axpy_kernel(T const alpha, T const *dx, T *dy, uint64_t const N) {
    uint64_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N) {
        dy[tid] = alpha * dx[tid] + dy[tid];
    }
}

// CUDA vector-vector addition
template <typename T>
void cuda_axpy(T const alpha, T const *hx, T *hy, uint64_t const N) {
    // Allocate memory on the device-side (GPU-side)
    T *dx, *dy;
    cudaMalloc(&dx, N * sizeof(T));
    cudaMalloc(&dy, N * sizeof(T));

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(dx, hx, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, N * sizeof(T), cudaMemcpyHostToDevice);

    uint64_t const block_size = 256;
    uint64_t const num_blocks = (N + block_size - 1) / block_size;
    // Launch the kernel on the GPU
    cuda_axpy_kernel<<<num_blocks, block_size>>>(alpha, dx, dy, N);
    cudaMemcpy(hy, dy, N * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(dx);
    cudaFree(dy);
}

// cuBLAS vector-vector addition
template <typename T>
void cublas_axpy(T const alpha, T const *hx, T *hy, uint64_t const N) {
    // Allocate memory on the device-side (GPU-side)
    T *dx, *dy;
    cudaMalloc(&dx, N * sizeof(T));
    cudaMalloc(&dy, N * sizeof(T));

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(dx, hx, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, N * sizeof(T), cudaMemcpyHostToDevice);

    // Create and initialize a new context
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // Launch the kernel on the GPU
    if constexpr (std::is_same<T, float>::value)
        cublasSaxpy(handle, N, &alpha, dx, 1, dy, 1);
    else if constexpr (std::is_same<T, double>::value)
        cublasDaxpy(handle, N, &alpha, dx, 1, dy, 1);
    else
        throw std::runtime_error("Unsupported type");
    cudaMemcpy(hy, dy, N * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(dx);
    cudaFree(dy);
    cublasDestroy_v2(handle);
}