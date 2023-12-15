#include <future>
#include <iostream>
#include <vector>

#include "common.h"

static int NSTEPS = 1'000'000'000;

int main() {
    double pi1 = 0, pi2 = 0;
    std::cout << "\nEstimate pi by integral: pi = 4 * int_0^1 dx / (1 + x^2)"
              << "\n";

    auto f = [](int nsteps) -> double {
        double dx = 1.0 / nsteps, pi = 0.0;
        for (int i = 0; i < nsteps; ++i) {
            double x = (i + 0.5) * dx;
            pi += 4.0 / (1.0 + x * x) * dx;
        }
        return pi;
    };

    {
        common::Timer timer("Normal  ");
        pi1 = f(NSTEPS);
    }

    size_t segmentSize = NSTEPS / 8;  // 8 threads
    std::vector<std::future<double>> futures;

    auto g = [](int nsteps, size_t start, size_t end) -> double {
        double dx = 1.0 / nsteps, pi = 0.0;
        for (size_t i = start; i < end; ++i) {
            double x = (i + 0.5) * dx;
            pi += 4.0 / (1.0 + x * x) * dx;
        }
        return pi;
    };

    {
        common::Timer timer("Parallel");
        for (size_t i = 0; i < 8; ++i) {
            size_t start = i * segmentSize;
            size_t end = (i == 7) ? NSTEPS : (i + 1) * segmentSize;
            futures.emplace_back(std::async(std::launch::async, g, NSTEPS, start, end));
        }
        for (auto& future : futures) pi2 += future.get();
    }

    std::cout << "diff(pi1, pi2): " << std::abs(pi1 - pi2) << "\n";

    return 0;
}
