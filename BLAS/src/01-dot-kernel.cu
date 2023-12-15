#include <cublas_v2.h>     // cublasHandle_t, cublasCreate_v2, cublasDestroy_v2, cublasSaxpy, cublasDaxpy
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, cudaFree

#include <cstdint>    // uint64_t
#include <stdexcept>  // std::runtime_error

template <typename T>
T cublas_dot(T const *ha, T const *hb, uint64_t const N) {
    // Allocate memory on the device-side (GPU-side)
    T *da, *db;

    cudaMalloc(&da, N * sizeof(T));
    cudaMalloc(&db, N * sizeof(T));

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(da, ha, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(T), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // Perform the dot product
    T res = 0;
    if constexpr (std::is_same<T, float>::value)
        cublasSdot(handle, N, da, 1, db, 1, &res);
    else if constexpr (std::is_same<T, double>::value)
        cublasDdot(handle, N, da, 1, db, 1, &res);
    else
        throw std::runtime_error("Unsupported type");

    // Free the device memory
    cudaFree(da);
    cudaFree(db);
    cublasDestroy_v2(handle);
    return res;
}