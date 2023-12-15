#include <atomic>
#include <future>
#include <iostream>
#include <vector>

#include "common.h"

int main() {
    std::cout << "\nCalculate dot product of two vectors"
              << "\n";
    size_t const N = 100'000'000;
    double res1 = 0, res2 = 0, res4 = 0;

    std::vector<double> x(N), y(N);
    srand(time(NULL));
    for (size_t i = 0; i < N; ++i) {
        x[i] = (double)rand() / RAND_MAX;
        y[i] = (double)rand() / RAND_MAX;
    }

    {
        common::Timer timer("Normal  ");
        for (size_t i = 0; i < x.size(); ++i) res1 += x[i] * y[i];
    }

    size_t segmentSize = N / 8;  // 8 threads
    std::vector<std::future<double>> futures;

    // This isn't good, should use std::reference_wrapper or pass std::vector<double> const& and use std::ref
    // in async
    auto f = [&x, &y](size_t start, size_t end) -> double {
        double res = 0;
        for (size_t i = start; i < end; ++i) res += x[i] * y[i];
        return res;
    };
    {
        common::Timer timer("Parallel");
        for (size_t i = 0; i < 8; ++i) {
            size_t start = i * segmentSize;
            size_t end = (i == 7) ? N : (i + 1) * segmentSize;
            futures.emplace_back(std::async(std::launch::async, f, start, end));
        }
        for (auto& future : futures) res2 += future.get();
    }

    // std::atomic to update result, just for reference
    std::atomic<double> res3 = 0;
    std::vector<std::future<void>> futs;

    std::function<void(size_t, size_t)> g = [&x, &y, &res3](size_t start, size_t end) {
        double res = 0;
        for (size_t i = start; i < end; ++i) res += x[i] * y[i];
        res3 += res;  // C++20
    };
    {
        common::Timer timer("Parallel");
        for (size_t i = 0; i < 8; ++i) {
            size_t start = i * segmentSize;
            size_t end = (i == 7) ? N : (i + 1) * segmentSize;
            futs.emplace_back(std::async(std::launch::async, g, start, end));
        }
        for (auto& future : futs) future.get();
    }

    futs.clear();
    std::mutex m;
    std::function<void(size_t, size_t)> h = [&x, &y, &res4, &m](size_t start, size_t end) {
        double r = 0;
        for (size_t i = start; i < end; ++i) r += x[i] * y[i];
        {
            std::lock_guard<std::mutex> lock(m);
            res4 += r;
        }
    };
    {
        common::Timer timer("Parallel");
        for (size_t i = 0; i < 8; ++i) {
            size_t start = i * segmentSize;
            size_t end = (i == 7) ? N : (i + 1) * segmentSize;
            futs.emplace_back(std::async(std::launch::async, h, start, end));
        }
        for (auto& future : futs) future.get();
    }

    std::cout << "diff(res1, res2): " << std::abs(res1 - res2) << "\n";
    std::cout << "diff(res1, res3): " << std::abs(res1 - res3) << "\n";
    std::cout << "diff(res1, res4): " << std::abs(res1 - res4) << "\n";

    return 0;
}
