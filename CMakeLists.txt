cmake_minimum_required(VERSION 3.17)

project(gpu_project LANGUAGES CUDA CXX)

find_package(CUDAToolkit)

add_executable(${PROJECT_NAME} cpu/dijkstras.cpp cpu/bellman.cpp cuda/bellman_kernels.cu cuda/bellman_kernels.cuh cuda/bellman.cu cuda/bellman.cuh cuda/dijkstras_kernels.cu cuda/dijkstras_kernels.cuh cuda/dijkstras.cu cuda/dijkstras.cuh main.h main.cpp)