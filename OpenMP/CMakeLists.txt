cmake_minimum_required(VERSION 3.12)
project(OpenMPExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror") # Optional
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find OpenMP package
find_package(OpenMP REQUIRED)
message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")
message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
message(STATUS "OpenMP CXX libraries: ${OpenMP_CXX_LIBRARIES}")

add_executable(dot src/dot.cpp src/common.h)
target_link_libraries(dot PRIVATE OpenMP::OpenMP_CXX)

add_executable(pi src/pi.cpp src/common.h)
target_link_libraries(pi PRIVATE OpenMP::OpenMP_CXX)

add_executable(fibonacci src/fibonacci.cpp src/common.h)
target_link_libraries(fibonacci PRIVATE OpenMP::OpenMP_CXX)

add_executable(conv2d src/conv2d.cpp src/common.h)
target_link_libraries(conv2d PRIVATE OpenMP::OpenMP_CXX)
