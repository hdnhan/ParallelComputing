#include <omp.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "common.h"

int main() {
    std::cout << "\nDot product\n";
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";
    const int N = 100'000'000;
    double sum1 = 0, sum2 = 0;
    std::vector<double> x(N), y(N);

    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        x[i] = (double)rand() / RAND_MAX;
        y[i] = (double)rand() / RAND_MAX;
    }

    {
        common::Timer timer("Normal  ");
        for (int i = 0; i < N; ++i) {
            sum1 += x[i] * y[i];
        }
    }

    {
        common::Timer timer("Parallel");
#pragma omp parallel for reduction(+ : sum2) schedule(static)
        for (int i = 0; i < N; ++i) {
            sum2 += x[i] * y[i];
        }
    }

    std::cout << "diff(sum1, sum2): " << std::abs(sum1 - sum2) << "\n";

    return 0;
}