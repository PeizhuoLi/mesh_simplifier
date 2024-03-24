cmake_minimum_required(VERSION 3.4...3.18)
project(mesh_simplifier)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)

add_subdirectory(pybind11)
add_subdirectory(libigl)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(libigl/include)
pybind11_add_module(mesh_simplifier src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(mesh_simplifier
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
