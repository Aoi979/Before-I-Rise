add_executable(fa_test main.cpp)

target_link_libraries(fa_test PRIVATE before_i_rise)
set(CUDA_HEADERS "/opt/cuda")
target_include_directories(fa_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(fa_test PRIVATE ${CUDA_HEADERS}/include)
