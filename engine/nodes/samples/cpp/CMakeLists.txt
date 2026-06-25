cmake_minimum_required(VERSION 3.16)
project(gaussian_blur_node)

set(CMAKE_CXX_STANDARD 17)

# ── Find dependencies ────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# ── CV-FLOW SDK headers ──────────────────────────────────────────────────────
# Adjust path if you copied the headers elsewhere
set(CVFLOW_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../shared/cpp/include"
    CACHE PATH "Path to CV-FLOW shared/cpp/include directory")

# ── Build mode ───────────────────────────────────────────────────────────────
# Default: sequential (ctypes .so)
# Set -DCVFLOW_MODE=multiprocess for standalone executable
set(CVFLOW_MODE "sequential" CACHE STRING "sequential | multiprocess")

if(CVFLOW_MODE STREQUAL "sequential")
    # Shared library loaded by Python engine via ctypes
    add_library(node SHARED gaussian_blur_node.cpp)
    target_include_directories(node PRIVATE ${CVFLOW_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(node PRIVATE ${OpenCV_LIBS})
    set_target_properties(node PROPERTIES
        PREFIX ""            # → node.so not libnode.so
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(node PRIVATE -O2 -march=native)
    message(STATUS "Building gaussian_blur_node as shared library (sequential mode)")

elseif(CVFLOW_MODE STREQUAL "multiprocess")
    # Standalone executable for multi-process mode
    add_executable(gaussian_blur_node gaussian_blur_node.cpp)
    target_include_directories(gaussian_blur_node PRIVATE ${CVFLOW_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(gaussian_blur_node PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(gaussian_blur_node PRIVATE CVFLOW_MULTIPROCESS)
    target_compile_options(gaussian_blur_node PRIVATE -O2 -march=native)
    message(STATUS "Building gaussian_blur_node as executable (multiprocess mode)")

else()
    message(FATAL_ERROR "Unknown CVFLOW_MODE: ${CVFLOW_MODE}. Use 'sequential' or 'multiprocess'")
endif()
