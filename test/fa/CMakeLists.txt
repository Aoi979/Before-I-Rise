add_executable(fa_test main.cpp)

target_link_libraries(fa_test PRIVATE before_i_rise)

target_include_directories(fa_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
