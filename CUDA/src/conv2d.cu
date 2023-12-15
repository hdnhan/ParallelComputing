#include <assert.h>
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, cudaFree

#include <iostream>
#include <vector>

#include "common.h"

// Convolution kernel
__global__ void conv2d_kernel(float const* input, float* output, float const* weight, float const* bias,
                              int N, int Ci, int Hi, int Wi, int Co, int Ho, int Wo, int ks1, int ks2,
                              int stride, int padding) {
    // ishape: [N, Ci, Hi, Wi]
    // oshape: [N, Co, Ho, Wo]

    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (c < Co && h < Ho && w < Wo) {
        int on = Co * Ho * Wo;
        int oc = Ho * Wo;
        int in = Ci * Hi * Wi;
        int ii = Hi * Wi;
        int wc = Ci * ks1 * ks2;
        int wi = ks1 * ks2;
        for (int n = 0; n < N; ++n) {
            for (int i = 0; i < Ci; ++i) {
                for (int j = 0; j < ks1; ++j) {
                    for (int k = 0; k < ks2; ++k) {
                        // [n, c, h, w] += [n, i, h * stride + j, w * stride + k] * [c, i, j, k]
                        output[n * on + c * oc + h * Wo + w] +=
                            input[n * in + i * ii + (h * stride + j) * Wi + (w * stride + k)] *
                            weight[c * wc + i * wi + j * ks2 + k];
                    }
                }
            }
            output[n * on + c * oc + h * Wo + w] += bias[c];
        }
    }
}

class Conv2d {
   private:
    int in_channels;
    int out_channels;
    std::pair<int, int> kernel_size;
    int stride;
    int padding;
    bool bias;

    // shape: [out_channels, in_channels, kernel_size.first, kernel_size.second]
    std::vector<float> _weight;
    // shape: [out_channels]
    std::vector<float> _bias;

   public:
    Conv2d(int in_channels, int out_channels, std::pair<int, int> kernel_size, int stride = 1,
           int padding = 0, bool bias = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          bias(bias) {
        // Validate the input arguments
        assert(in_channels > 0);
        assert(out_channels > 0);
        assert(kernel_size.first > 0 && kernel_size.second > 0);
        assert(stride > 0);
        assert(padding >= 0);

        // Initialize the weights and bias
        _weight = std::vector<float>(out_channels * in_channels * kernel_size.first * kernel_size.second);
        _bias = std::vector<float>(out_channels);
        set_random_weights();
    }

    void set_random_weights() {
        srand(time(NULL));
        if (bias)
            for (uint32_t i = 0; i < _bias.size(); ++i) _bias[i] = (float)rand() / RAND_MAX;
        for (uint32_t i = 0; i < _weight.size(); ++i) _weight[i] = (float)rand() / RAND_MAX;
    }

    void set_weights(std::vector<float> const& weight, std::vector<float> const& bias) {
        _weight = weight;
        _bias = bias;
    }

    std::pair<std::vector<int>, std::vector<float>> forward(
        std::pair<std::vector<int>, std::vector<float>> const& input) {
        // auto [ishape, idata] = input;
        std::vector<int> ishape = input.first;
        std::vector<float> idata = input.second;
        // input shape: [N, Ci, Hi, Wi]
        // output shape: [N, Co, Ho, Wo]
        int N = ishape[0];
        int Hi = ishape[2];
        int Wi = ishape[3];

        // Validate the input arguments
        assert(ishape[1] == in_channels);

        int Ho = (Hi + 2 * padding - kernel_size.first) / stride + 1;
        int Wo = (Wi + 2 * padding - kernel_size.second) / stride + 1;
        std::vector<int> oshape = {N, out_channels, Ho, Wo};
        std::vector<float> odata = std::vector<float>(N * out_channels * Ho * Wo, 0.0);

        // Allocate memory on the device
        float *d_idata, *d_odata, *d_weight, *d_bias;
        cudaMalloc(&d_idata, N * in_channels * Hi * Wi * sizeof(float));
        cudaMalloc(&d_odata, N * out_channels * Ho * Wo * sizeof(float));
        cudaMalloc(&d_weight,
                   out_channels * in_channels * kernel_size.first * kernel_size.second * sizeof(float));
        cudaMalloc(&d_bias, out_channels * sizeof(float));

        // Copy the data to the device
        cudaMemcpy(d_idata, idata.data(), N * in_channels * Hi * Wi * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, _weight.data(),
                   out_channels * in_channels * kernel_size.first * kernel_size.second * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, _bias.data(), out_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_odata, odata.data(), N * out_channels * Ho * Wo * sizeof(float), cudaMemcpyHostToDevice);

        // CUDA kernel
        // Threads per CTA dimension
        uint64_t const XTHREADS = 32;
        uint64_t const YTHREADS = 32;
        uint64_t const ZTHREADS = 1;

        // CTAs per grid dimension
        uint64_t const XCTAS = (Ho + XTHREADS - 1) / XTHREADS;
        uint64_t const YCTAS = (Wo + YTHREADS - 1) / YTHREADS;
        uint64_t const ZCTAS = (out_channels + ZTHREADS - 1) / ZTHREADS;

        dim3 const threads(XTHREADS, YTHREADS, ZTHREADS);
        dim3 const blocks(XCTAS, YCTAS, ZCTAS);

        // Launch the kernel
        conv2d_kernel<<<blocks, threads>>>(d_idata, d_odata, d_weight, d_bias, N, in_channels, Hi, Wi,
                                           out_channels, Ho, Wo, kernel_size.first, kernel_size.second,
                                           stride, padding);

        // Copy the output back to the host
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) std::cout << "Error: " << cudaGetErrorString(err) << "\n";
        cudaMemcpy(odata.data(), d_odata, N * out_channels * Ho * Wo * sizeof(float), cudaMemcpyDeviceToHost);

        // Free the device memory
        cudaFree(d_idata);
        cudaFree(d_odata);
        cudaFree(d_weight);
        cudaFree(d_bias);

        return {oshape, odata};
    }
};

int main() {
    std::cout << "\n2D Convolution\n";

    // Create an instance of the Conv2d class
    int N = 2, Ci = 3, Hi = 1024, Wi = 1024;
    Conv2d conv(3, 64, {3, 3});

    // Create a sample input tensor
    std::vector<int> ishape = {N, Ci, Hi, Wi};
    std::vector<float> idata(N * Ci * Hi * Wi, 1.0);

    // Perform the forward pass
    {
        common::Timer timer("Forward");
        auto output = conv.forward({ishape, idata});
    }

    return 0;
}