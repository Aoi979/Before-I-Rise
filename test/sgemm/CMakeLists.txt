add_executable(test main.cpp)

target_link_libraries(test PRIVATE before_i_rise)

target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR}/include)
