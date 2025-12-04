cmake_minimum_required(VERSION 4.0)
project(before_i_rise LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

file(GLOB_RECURSE BEFORE_I_RISE_CU
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(before_i_rise STATIC ${BEFORE_I_RISE_CU})

set_target_properties(before_i_rise PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(before_i_rise PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

add_subdirectory(test)