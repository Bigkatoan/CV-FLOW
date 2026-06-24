cmake_minimum_required(VERSION 3.16)
project(blur_node CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_library(node SHARED node.cpp)
target_include_directories(node PRIVATE
    ../../include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(node PRIVATE ${OpenCV_LIBS})
set_target_properties(node PROPERTIES PREFIX "")
target_compile_options(node PRIVATE -O2 -march=native)
