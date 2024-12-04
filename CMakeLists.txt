cmake_minimum_required(VERSION 3.20)

project(CudaEntropy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Найти CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Основной исполняемый файл
add_executable(CudaEntropy main.cu)

add_library(CudaLib SHARED lib.cu lib.cuh)
target_include_directories(CudaLib PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(CudaLib PRIVATE CUDA::cudart)


target_link_libraries(CudaEntropy PRIVATE CudaLib CUDA::cudart)
