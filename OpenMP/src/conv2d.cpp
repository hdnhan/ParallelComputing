#include <assert.h>
#include <omp.h>

#include <iostream>
#include <vector>

#include "common.h"

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
        auto [ishape, idata] = input;
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

        auto f = [&](int n, int c, int h, int w) {
            int on = out_channels * Ho * Wo;
            int oc = Ho * Wo;
            int in = in_channels * Hi * Wi;
            int ii = Hi * Wi;
            int wc = in_channels * kernel_size.first * kernel_size.second;
            int wi = kernel_size.first * kernel_size.second;
            for (int i = 0; i < in_channels; ++i) {
                for (int j = 0; j < kernel_size.first; ++j) {
                    for (int k = 0; k < kernel_size.second; ++k) {
                        // [n, c, h, w] += [n, i, h * stride + j, w * stride + k] * [c, i, j, k]
                        odata[n * on + c * oc + h * Wo + w] +=
                            idata[n * in + i * ii + (h * stride + j) * Wi + (w * stride + k)] *
                            _weight[c * wc + i * wi + j * kernel_size.second + k];
                    }
                }
            }
            odata[n * on + c * oc + h * Wo + w] += _bias[c];
        };

// Perform the convolution
#pragma omp parallel for collapse(4)
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < out_channels; ++c) {
                for (int h = 0; h < Ho; ++h) {
                    for (int w = 0; w < Wo; ++w) {
                        f(n, c, h, w);
                    }
                }
            }
        }
        return {oshape, odata};
    }
};

int main() {
    std::cout << "\n2D Convolution\n";
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";

    // Create an instance of the Conv2d class
    int N = 2, Ci = 3, Hi = 1024, Wi = 1024;
    Conv2d conv(3, 64, {3, 3});

    // Create a sample input tensor
    std::vector<int> ishape = {N, Ci, Hi, Wi};
    std::vector<float> idata(N * Ci * Hi * Wi, 1.0);

    // Perform the forward pass
    {
        common::Timer timer("Forward");
        auto [oshape, odata] = conv.forward({ishape, idata});
    }

    return 0;
}

/*
Results:
- [N, C, H, W] => 4s
- [N, H, W, C] + collapse(4) => 1.3s
- 1D array of size [N * C * H * W] => 3.2s
- 1D array of size [N * H * W * C] + collapse(4) => 0.9s
*/