
cmake_minimum_required(VERSION 3.20)  
project(MyCUDAProject LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
# Добавьте исходные файлы
add_executable(CudaEntropy main.cu lib.cuh)
