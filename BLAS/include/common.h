#pragma once

#include <chrono>    // std::chrono::high_resolution_clock
#include <cstdint>   // uint64_t
#include <iostream>  // std::cout
#include <string>    // std::string

namespace common {

/*
Timer class to measure the execution time of a block of code
*/
class Timer {
   public:
    Timer(std::string const &message);
    ~Timer();

   private:
    void stop();

   private:
    std::string message;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
};

/*
Initialize a vector with random values
@param a: pointer to the vector
@param N: size of the vector
@param low: lower bound of the random values
@param high: upper bound of the random values
*/
template <typename T>
void init(T *a, uint64_t const &N, T const &low, T const &high);

/*
Initialize a matrix with random values
@param a: pointer to the matrix
@param rows: number of rows
@param cols: number of columns
@param low: lower bound of the random values
@param high: upper bound of the random values
*/
template <typename T>
void init(T *A, uint64_t const &rows, uint64_t const &cols, T const &low, T const &high);

/*
Calculate the difference between two vectors
@param a: pointer to the first vector
@param b: pointer to the second vector
@param N: size of the vectors
*/
template <typename T>
T diff(T const *a, T const *b, uint64_t const &N);

/*
Calculate the difference between two matrices
@param A: pointer to the first matrix
@param B: pointer to the second matrix
@param rows: number of rows
@param cols: number of columns
*/
template <typename T>
T diff(T const *A, T const *B, uint64_t const &rows, uint64_t const &cols);

}  // namespace common