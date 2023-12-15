#include <omp.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "common.h"

uint64_t fibonacci(uint64_t n) {
    if (n < 2) return n;
    uint64_t x, y;
    x = fibonacci(n - 1);
    y = fibonacci(n - 2);
    return x + y;
}

uint64_t fibonacci_array(uint64_t n) {
    std::vector<uint64_t> fib(n + 1);
    fib[0] = 0;
    fib[1] = 1;
    for (uint64_t i = 2; i <= n; ++i) fib[i] = fib[i - 1] + fib[i - 2];
    return fib[n];
}

uint64_t fibonacci_task(uint64_t n) {
    if (n < 2) return n;
    uint64_t x, y;
#pragma omp task shared(x) if (n > 25)
    x = fibonacci_task(n - 1);
#pragma omp task shared(y) if (n > 25)
    y = fibonacci_task(n - 2);
#pragma omp taskwait
    return x + y;
}

// This is actually sequential because of the ordered clause
uint64_t fibonacci_ordered(uint64_t n) {
    std::vector<uint64_t> fib(n + 1);
    fib[0] = 0;
    fib[1] = 1;
#pragma omp parallel for ordered
    for (uint64_t i = 2; i <= n; ++i)
#pragma omp ordered
        fib[i] = fib[i - 1] + fib[i - 2];
    return fib[n];
}

int main() {
    uint64_t res1, res2, res3, res4;
    uint64_t n = 40;
    std::cout << "\nFibonacci(" << n << ")\n";
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";

    {
        common::Timer timer("Normal ");
        res1 = fibonacci(n);
    }

    {
        common::Timer timer("Array  ");
        res2 = fibonacci_array(n);
    }

    {
        common::Timer timer("Task   ");
#pragma omp parallel
#pragma omp single
#pragma omp taskgroup
        res3 = fibonacci_task(n);
    }

    {
        common::Timer timer("Ordered");
        res4 = fibonacci_ordered(n);
    }

    std::cout << "diff(res1, res2): " << res1 - res2 << "\n";
    std::cout << "diff(res1, res3): " << res1 - res3 << "\n";
    std::cout << "diff(res1, res4): " << res1 - res4 << "\n";

    return 0;
}