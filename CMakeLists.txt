cmake_minimum_required(VERSION 4.0)
project(before_i_rise CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
add_library(before_i_rise STATIC src/gemm/gemm.cu)
set_target_properties(before_i_rise PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
add_executable(test test/main.cpp)
target_link_libraries(test PRIVATE before_i_rise)
target_include_directories(test PRIVATE include)

