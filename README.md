### Вычисление гистограммы и энтропии на GPU с использованием CUDA
#### Описание

Данный проект реализует вычисление гистограммы и нормализованной энтропии для дискретной системы в трехмерном параметрическом пространстве. Основной расчет выполняется на GPU с использованием технологии CUDA, что позволяет значительно ускорить вычисления для больших объемов данных.

#### Доп информация
Все методы в lib.cuh дополнены комментариями. А также есть небольшое послесловие в конце файла с некоторыми моментами о реализации

#### Установка проекта с использованием CMake
Убедитесь, что на вашем компьютере установлены следующие компоненты:

- CUDA Toolkit (включая компилятор NVCC);
- CMake;
- Компилятор с поддержкой C++ (например, GCC или MSVC).

#### Сборка и запуск
1. Сначала склонируйте репозиторий проекта с помощью Git:
```bash
git clone https://github.com/Menoitami/CudaEntropy.git
cd CudaEntropy
```
Если вы скачивайте .zip, нужно его разархировать и зайти с терминала в папку, где будут лежать .cu и .cuh файлы, при помощи команды cd.
Далее все по инструкции.

2. Создание папки для сборки
Создание папки и заход в папку, где будут сгенерированные файлы.
```bash
mkdir build
cd build
```
3. Генерация файлов сборки
```bash
cmake ..
```
4. Сборка проекта
```bash
cmake --build .
```
5. Запуск программы
```bash
cd Debug
./CudaEntropy.exe <имя результирующего файла>
```
или 
```bash
./Debug/CudaEntropy.exe <имя результирующего файла>
```
