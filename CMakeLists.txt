# cmake_minimum_required(VERSION 3.10)

# # Указываем название проекта и его версию
# project(Entropy VERSION 1.0)

# # Настраиваем стандарт C++
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# # Указываем пути к FFTW
# set(FFTW_DIR "D:/libs/fftw-3.3.5-dll64") # Путь к заголовочным файлам


# # Добавляем пути для компилятора
# include_directories(${FFTW_DIR})
# link_directories(${FFTW_DIR})

# # Создаем исполняемый файл
# add_executable(Entropy main.cpp)

# # Линкуем библиотеку FFTW
# target_link_libraries(Entropy PRIVATE fftw3-3)

cmake_minimum_required(VERSION 3.20)  # Убедитесь, что версия CMake поддерживает CUDA
project(MyCUDAProject LANGUAGES CXX CUDA)

# Укажите минимальную версию C++ и CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Добавьте исходные файлы
add_executable(CudaEntropy main.cu lib.cuh)
