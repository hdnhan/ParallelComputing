cmake_minimum_required(VERSION 3.12)
project(BLASExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror") # Optional
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDA)
if (CUDA_FOUND)
    enable_language(CUDA)
endif()

message("CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message("CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message("CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message("CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

# Find OpenMP package
find_package(OpenMP REQUIRED)
message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")
message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
message(STATUS "OpenMP CXX libraries: ${OpenMP_CXX_LIBRARIES}")

# Find OpenBLAS
find_path(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS "/usr/include/x86_64-linux-gnu/openblas-openmp")
find_library(OpenBLAS_LIB NAMES openblas PATHS "/usr/lib/x86_64-linux-gnu/openblas-openmp")

add_library(common SHARED src/common.cpp)
target_include_directories(common PUBLIC include)

set (targets 01-axpy 01-dot 02-gemv 03-gemm)
foreach (target ${targets})
    if (CUDA_FOUND)
        cuda_add_executable(${target} src/${target}.cpp)
        set_source_files_properties(src/${target}.cpp PROPERTIES LANGUAGE CUDA)
        target_compile_definitions(${target} PRIVATE CUDA_FOUND)
        target_link_libraries(${target} cublas)
    else()
        add_executable(${target} src/${target}.cpp)
    endif()
    target_link_libraries(${target} common ${OpenBLAS_LIB} OpenMP::OpenMP_CXX)
endforeach()
