add_executable(test main.cpp)

target_link_libraries(test PRIVATE before_i_rise)

set(CUDA_HEADERS "/opt/cuda")
target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test PRIVATE ${CUDA_HEADERS}/include)
