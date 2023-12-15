#include <omp.h>

#include <iostream>
#include <vector>

#include "common.h"

static int NSTEPS = 1'000'000'000;

double pi_cacl(int nsteps) {
    double dx = 1.0 / nsteps, pi = 0.0;
    for (int i = 0; i < nsteps; ++i) {
        double x = (i + 0.5) * dx;
        pi += 4.0 / (1.0 + x * x) * dx;
    }
    return pi;
}

// Problem: false sharing
double pi_parallel(int nsteps) {
    double dx = 1.0 / nsteps, pi = 0.0;
    int nthreads;
    std::vector<double> sum(omp_get_max_threads(), 0.0);
#pragma omp parallel default(shared)
    {
        // Only one thread updates nthreads
#pragma omp master
        { nthreads = omp_get_num_threads(); }

        int id = omp_get_thread_num();
        for (int i = id; i < nsteps; i += omp_get_num_threads()) {
            double x = (i + 0.5) * dx;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (int i = 0; i < nthreads; ++i) pi += sum[i] * dx;
    return pi;
}

// Solution: use atomic
double pi_parallel2(int nsteps) {
    double dx = 1.0 / nsteps, pi = 0.0;
#pragma omp parallel default(shared)
    {
        double sum = 0.0;
        int id = omp_get_thread_num();
        for (int i = id; i < nsteps; i += omp_get_num_threads()) {
            double x = (i + 0.5) * dx;
            sum += 4.0 / (1.0 + x * x);
        }
#pragma omp atomic
        pi += sum * dx;
    }
    return pi;
}

// Solution: use reduction
double pi_for(int nsteps) {
    double dx = 1.0 / nsteps, pi = 0.0;
#pragma omp parallel for reduction(+ : pi) schedule(static)
    for (int i = 0; i < nsteps; ++i) {
        double x = (i + 0.5) * dx;
        pi += 4.0 / (1.0 + x * x) * dx;
    }
    return pi;
}

int main() {
    double pi1, pi2, pi3, pi4;
    std::cout << "\nEstimate pi by integral: pi = 4 * int_0^1 dx / (1 + x^2)"
              << "\n";
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";

    {
        common::Timer timer("Normal  ");
        pi1 = pi_cacl(NSTEPS);
    }
    {
        common::Timer timer("Parallel");
        pi2 = pi_parallel(NSTEPS);
    }

    {
        common::Timer timer("Parallel");
        pi3 = pi_parallel2(NSTEPS);
    }

    {
        common::Timer timer("For     ");
        pi4 = pi_for(NSTEPS);
    }

    std::cout << "diff(pi1, pi2): " << std::abs(pi1 - pi2) << "\n";
    std::cout << "diff(pi1, pi3): " << std::abs(pi1 - pi3) << "\n";
    std::cout << "diff(pi1, pi4): " << std::abs(pi1 - pi4) << "\n";

    return 0;
}
