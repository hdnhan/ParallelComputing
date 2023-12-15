#include "common.h"

namespace common {

Timer::Timer(std::string const &message) : message(message) {
    start = std::chrono::high_resolution_clock::now();
}
Timer::~Timer() { stop(); }
void Timer::stop() {
    end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << message << ": " << 1e-6 * duration << "s (" << 1e-3 * duration << "ms)"
              << "\n";
}

template <typename T>
void init(T *a, uint64_t const &N, T const &low, T const &high) {
    srand((uint64_t)time(NULL));
    for (uint64_t i = 0; i < N; ++i) a[i] = (T)rand() / (T)RAND_MAX * (high - low) + low;
}
template void init(float *, uint64_t const &, float const &, float const &);
template void init(double *, uint64_t const &, double const &, double const &);

template <typename T>
void init(T *A, uint64_t const &rows, uint64_t const &cols, T const &low, T const &high) {
    srand((uint64_t)time(NULL));
    for (uint64_t i = 0; i < rows; ++i)
        for (uint64_t j = 0; j < cols; ++j) A[i * cols + j] = (T)rand() / (T)RAND_MAX * (high - low) + low;
}
template void init(float *, uint64_t const &, uint64_t const &, float const &, float const &);
template void init(double *, uint64_t const &, uint64_t const &, double const &, double const &);

template <typename T>
T diff(T const *a, T const *b, uint64_t const &N) {
    T max_diff = 0;
    for (uint64_t i = 0; i < N; ++i) max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    return max_diff;
}
template float diff(float const *, float const *, uint64_t const &);
template double diff(double const *, double const *, uint64_t const &);

template <typename T>
T diff(T const *A, T const *B, uint64_t const &rows, uint64_t const &cols) {
    T max_diff = 0;
    for (uint64_t i = 0; i < rows; ++i)
        for (uint64_t j = 0; j < cols; ++j)
            max_diff = std::max(max_diff, std::abs(A[i * cols + j] - B[i * cols + j]));
    return max_diff;
}
template float diff(float const *, float const *, uint64_t const &, uint64_t const &);
template double diff(double const *, double const *, uint64_t const &, uint64_t const &);

}  // namespace common