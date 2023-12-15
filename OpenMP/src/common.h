#pragma once

#include <chrono>    // std::chrono::high_resolution_clock
#include <iostream>  // std::cout
#include <string>    // std::string

namespace common {

/*
Timer class to measure the execution time of a block of code
*/
class Timer {
   public:
    Timer(std::string const &message) : message(message) {
        start = std::chrono::high_resolution_clock::now();
    }
    ~Timer() { stop(); }

   private:
    void stop() {
        end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << message << ": " << 1e-6 * duration << "s (" << 1e-3 * duration << "ms)"
                  << "\n";
    }

   private:
    std::string message;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
};

}  // namespace common