add_executable(hgemm_test main.cpp)

target_link_libraries(hgemm_test PRIVATE before_i_rise)

target_include_directories(hgemm_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
