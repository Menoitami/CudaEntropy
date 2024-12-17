#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <thread>
#include <nvrtc.h>


#define CHECK_CUDA_ERROR(call)                                              \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)          \
                      << " in file " << __FILE__ << " at line " << __LINE__ \
                      << std::endl;                                         \
            exit(err);                                                      \
        }                                                                   \
    }

// Необходимые переменные, что добавляются в константную память видеокарты
__constant__ int d_histEntropySize; // Общий размер массива энтропии
__constant__ int d_histEntropySizeRow; // Строки массива энтропии
__constant__ int d_histEntropySizeCol; // Столбцы массива энтропии
__constant__ double d_startBin; // Начало диапазона гистогроаммы
__constant__ double d_endBin; // Конец диапазона гистограммы
__constant__ double d_stepBin; // Шаг гистограммы
__constant__ double d_tMax; // Время моделирования системы
__constant__ double d_transTime; // Время прогона системы (transient time)
__constant__ double d_h; // шаг системы
__constant__ int d_coord; // Индекс координаты, для которой ищется энтропия
__constant__ int d_paramsSize; // Колличество параметров системы
__constant__ int d_XSize; // Колличество координат системы
__constant__ int d_paramNumberA; // Индекс первого параметра
__constant__ int d_paramNumberB; // Индекс второго параметра
__device__ int d_progress; // Подсчет прогресса
__device__ const double EPSD = std::numeric_limits<double>::epsilon();

/**
 * @brief Создает равномерно распределенные точки в заданном интервале.
 * 
 * @param start Начало интервала.
 * @param end Конец интервала.
 * @param num Количество точек (должно быть >= 0).
 * 
 * @return std::vector<double> Вектор с равномерно распределенными точками.
 * 
 * @throws std::invalid_argument Если `num < 0`.
 */
__host__ std::vector<double> linspaceNum(double start, double end, int num)
{
    std::vector<double> result;

    if (num < 0)
        throw std::invalid_argument("received negative number of points");
    if (num == 0)
        return result;
    if (num == 1)
    {
        result.push_back(start);
        return result;
    }

    double step = (end - start) / (num - 1);

    for (int i = 0; i < num; ++i)
    {
        result.push_back(start + i * step);
    }

    return result;
}

/**
 * @brief Записывает 2D-вектор данных в CSV-файл.
 * 
 * @param histEntropy2D Двумерный вектор данных для записи.
 * @param cols Количество столбцов в данных.
 * @param rows Количество строк в данных.
 * @param filename Имя выходного CSV-файла.
 * 
 * @details Записывает данные в формате CSV, разделяя элементы строк запятыми.
 * Если файл не удается открыть, выводит сообщение об ошибке.
 */
__host__ void writeToCSV(const std::vector<std::vector<double>> &histEntropy2D, int cols, int rows, const std::string &filename)
{
    std::ofstream outFile(filename);

    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            outFile << histEntropy2D[i][j];
            if (j < cols - 1)
            {
                outFile << ",";
            }
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "Data successfully written to file " << filename << std::endl;
}

/**
 * @brief Преобразует одномерный вектор в двумерный.
 * 
 * @param histEntropy1D Одномерный вектор данных.
 * 
 * @return std::vector<std::vector<double>> Двумерный вектор, где исходный вектор 
 * становится единственной строкой.
 */
std::vector<std::vector<double>> convert1DTo2D(const std::vector<double> &histEntropy1D)
{
    return {histEntropy1D};
}

/**
 * @brief Записывает 1D-вектор данных в CSV-файл.
 * 
 * @param histEntropy1D Одномерный вектор данных для записи.
 * @param cols Количество столбцов в данных.
 * @param filename Имя выходного CSV-файла.
 * 
 * @details Преобразует 1D-вектор в 2D-вектор с одной строкой и записывает в CSV-файл.
 * Если файл не удается открыть, выводит сообщение об ошибке.
 */
__host__ void writeToCSV(const std::vector<double> &histEntropy1D, int cols, const std::string &filename)
{
    auto histEntropy2D = convert1DTo2D(histEntropy1D);
    writeToCSV(histEntropy2D, cols, 1, filename);
}

/**
 * @brief Вычисляет дискретную модель динамической системы.
 * 
 * @param x Массив из X_size элементов, представляющий состояние системы. Массив изменяется на месте.
 * @param a Массив из param_size параметров, определяющих характеристики системы.
 * @param h Шаг интегрирования.
 * 
 * В качестве системы может быть использована любая другая, достаточно поменять реализацию
 */
__device__ __host__ void calculateDiscreteModel(double *x, const double *a, const double h)
{
    double h1 = 0.5 * h + a[0];
    double h2 = 0.5 * h - a[0];

    
    x[0] += h1 * (-x[1] - x[2]);
    x[1] += h1 * (x[0] + a[1] * x[1]);
    x[2] += h1 * (a[2] + x[2] * (x[0] - a[3]));

    x[2] = (x[2] + h2 * a[2]) / (1 - h2 * (x[0] - a[3]));
    x[1] = (x[1] + h2 * x[0]) / (1 - h2 * a[1]);
    x[0] += h2 * (-x[1] - x[2]);
/*
    Пример системы из наших лабораторных

    double h1 = 0.5 * h + a[0];
    double h2 = 0.5 * h - a[0];

    x[0] += h1 * x[1];
    x[1] += h1 * x[2];
    x[2] += h1 * (x[1] * x[3] - a[1] * x[2] * x[3]);
    x[3] += h1 * (a[2] * x[0] * x[2] - x[1] * x[1] + x[2] * x[2]);

    x[3] += h2 * (a[2] * x[0] * x[2] - x[1] * x[1] + x[2] * x[2]);
    x[2] = (x[2] + h2 * x[1] * x[3]) / (1 + a[1] * h2 * x[3]);
    x[1] += h2 * x[2];
    x[0] += h2 * x[1];
*/
}

/**
 * @brief Выполняет многократные итерации вычисления дискретной модели системы.
 * 
 * @param x Массив из x_size элементов, представляющий текущее состояние системы. Массив изменяется на месте.
 * @param values Массив параметров модели.
 * @param h Шаг интегрирования.
 * @param amountOfIterations Количество итераций основного цикла.
 * @param preScaller Количество итераций внутреннего цикла перед записью данных.
 * @param writableVar Индекс переменной  x , значение которой записывается в массив `data`.
 * @param maxValue Максимальное допустимое значение для переменной  x[writableVar] . 
 * @param data Массив для записи значений переменной  x[writableVar]  на каждой итерации.
 * @param startDataIndex Начальный индекс в массиве `data` для записи значений.
 * @param writeStep Шаг записи в массиве `data` между значениями.
 * 
 * @return `true`, если максимальное значение не превышено; `false` в противном случае.
 */
__device__ __host__ bool loopCalculateDiscreteModel(double *x, const double *values, const double h,
                                                    const int amountOfIterations, const int preScaller,
                                                    int writableVar, const double maxValue, double *data, const int startDataIndex, const int writeStep)
{
    bool exceededMaxValue = (maxValue != 0);
    for (int i = 0; i < amountOfIterations; ++i)
    {
        if (data != nullptr)
            data[startDataIndex + i * writeStep] = x[writableVar];

        for (int j = 0; j < preScaller - 1; ++j)
            calculateDiscreteModel(x, values, h);

        calculateDiscreteModel(x, values, h);

        if (exceededMaxValue && fabsf(x[writableVar]) > maxValue)
            return false;
    }
    return true;
}

/**
 * @brief Вычисляет энтропию по заданным данным в гистограмме.
 * 
 * @param bins Массив значений гистограммы.
 * @param binSize Размер массива `bins`.
 * @param sum Сумма всех элементов в массиве `bins`.
 * 
 * @return double Значение энтропии.
 */
__device__ double calculateEntropy(double *bins, int binSize, const int sum)
{
    double entropy = 0.0;

    for (int i = 0; i < binSize; ++i)
    {
        if (bins[i] > 0)
        {
            bins[i] = (bins[i] / (sum + EPSD));
            entropy -= (bins[i] - EPSD) * log2(bins[i]);
        }
    }

    return entropy;
}

/**
 * @brief Рассчитывает гистограмму на основе изменения значения координаты системы во времени.
 * 
 * @param X Массив из X_size элементов, представляющий текущее состояние системы.
 * @param param Массив параметров модели.
 * @param sum Переменная для хранения общей суммы всех попаданий в bins. Изменяется на месте.
 * @param bins Массив для хранения значений гистограммы.
 */
__device__ void CalculateHistogram(
    double *X, const double *param,
    int &sum, double *bins)
{

    int iterations = static_cast<int>(d_tMax / d_h);
    double last = X[d_coord];
    bool lastBigger = false;

    int binSize = static_cast<int>(ceil((d_endBin - d_startBin) / d_stepBin));

    for (int i = 0; i < iterations; ++i)
    {
        calculateDiscreteModel(X, param, d_h);

        if (X[d_coord] > last)
        {
            lastBigger = true;
        }
        else if (X[d_coord] < last && lastBigger)
        {
            if (last >= d_startBin && last < d_endBin)
            {

                int index = static_cast<int>((last - d_startBin) / d_stepBin);
                if (index < binSize && index >= 0)
                {
                    bins[index]++;
                }
                sum++;
            }
            lastBigger = false;
        }
        last = X[d_coord];
    }
}

/**
 * @brief Рассчитывает гистограмму и энтропию для системы в трехмерном параметрическом пространстве на GPU.
 * 
 * @param X Указатель на массив начальных значений координат системы (размер XSize).
 * @param params Указатель на массив параметров системы (размер paramsSize).
 * @param paramLinspaceA Указатель на массив значений первого параметра (размер histEntropySizeRow).
 * @param paramLinspaceB Указатель на массив значений второго параметра (размер histEntropySizeCol).
 * @param histEntropy Указатель на массив для записи нормализованных значений энтропии (размер histEntropySize).
 * @param bins_global Указатель на глобальную память для хранения гистограмм (размер histEntropySize * binSize).
 * 
 * ### Алгоритм:
 * 
 * 1. Вычисление индекса потока  idx и привязка к точке параметрического пространства.
 * 
 * 2. Копирование входных данных (`X` и `params`) в локальную память для ускорения вычислений.
 * 
 * 3. Настройка параметров системы, используя сетки paramLinspaceA и paramLinspaceB.
 * 
 * 4. Стабилизация системы методом `loopCalculateDiscreteModel` на времени d_transTime \f$.
 * 
 * 5. Построение гистограммы методом `CalculateHistogram`.
 * 
 * 6. Вычисление энтропии методом `calculateEntropy` и сохранение результата.
 */
__global__ void calculateHistEntropyCuda3D(const double *X,
                                           const double *params,
                                           const double *paramLinspaceA,
                                           const double *paramLinspaceB,
                                           double *histEntropy,
                                           double *bins_global)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_histEntropySize)
        return;

    int binSize = static_cast<int>(std::ceil((d_endBin - d_startBin) / d_stepBin));
    int offset = idx * binSize;
    if (offset + binSize > d_histEntropySize * binSize)
        return;
    double *bin = &bins_global[offset];

    int row = idx / d_histEntropySizeCol;
    int col = idx % d_histEntropySizeCol;
    
    // Влияет на скорость работы, поставили прозапас, но нужно изменять размеры под актуальное решение! (без агрессии, просто обратить внимание)
    double X_locals[32];
    double params_local[32];

    memcpy(X_locals, X, d_XSize * sizeof(double));
    memcpy(params_local, params, d_paramsSize * sizeof(double));

    params_local[d_paramNumberA] = paramLinspaceA[row];
    params_local[d_paramNumberB] = paramLinspaceB[col];


    loopCalculateDiscreteModel(X_locals, params_local, d_h,
                               static_cast<int>(d_transTime / d_h),
                               0, 0, 0, nullptr, 0, 0);

    int sum = 0;

    CalculateHistogram(X_locals, params_local, sum, bin);
    double H = calculateEntropy(bin, binSize, sum);

    histEntropy[row * d_histEntropySizeCol + col] = (H / __log2f(binSize));

    // Подсчет прогресса выполнения задачи видеокартой 
    if ((atomicAdd(&d_progress, 1) + 1) % (d_histEntropySize / 10) == 0)
    {
        printf("Progress: %d%%\n", ((d_progress + 1) * 100) / d_histEntropySize);
    }
}

/**
 * @brief Функция для вычисления энтропии по 2 параметрам с использованием CUDA.
 * 
 * Метод выполняет вычисления энтропии гистограммы для данных, представленных в виде двух массивов параметров.
 * Он распределяет данные на несколько блоков и выполняет параллельные вычисления на GPU для оптимизации времени.
 * 
 * @param transTime Время транзиенты.
 * @param tMax Время работы системы.
 * @param h Шаг.
 * @param X координаты системы.
 * @param coord Индекс координаты по которой будет строится энтропия.
 * @param params Вектор параметров, где params[0] - коэффициент симметрии.
 * @param paramNumberA Индекс первого параметра для вычислений в params.
 * @param paramNumberB Индекс второго параметра для вычислений в params.
 * @param startBin Начало диапазона для столбцов гистограмм.
 * @param endBin Конец диапазона для столбцов гистограмм.
 * @param stepBin Шаг столбцов гистограмм.
 * @param linspaceStartA Начало диапазона для массива A.
 * @param linspaceEndA Конец диапазона для массива A.
 * @param linspaceNumA Количество точек в массиве A.
 * @param linspaceStartB Начало диапазона для массива B.
 * @param linspaceEndB Конец диапазона для массива B.
 * @param linspaceNumB Количество точек в массиве B.
 * 
 * @return Вектор двухмерный массив значений.
 */
__host__ std::vector<std::vector<double>> histEntropyCUDA3D(
    const double transTime, const double tMax, const double h,
    const std::vector<double> &X, const int coord,
    const std::vector<double> &params, const int paramNumberA, const int paramNumberB,
    const double startBin, const double endBin, const double stepBin,
    double linspaceStartA, double linspaceEndA, int linspaceNumA, double linspaceStartB, double linspaceEndB, int linspaceNumB)
{
    //Проверка на корректность введеных данных
    try
    {
        if (tMax <= 0)
            throw std::invalid_argument("tMax <= 0");
        if (transTime <= 0)
            throw std::invalid_argument("transTime <= 0");
        if (h <= 0)
            throw std::invalid_argument("h <= 0");
        if (startBin >= endBin)
            throw std::invalid_argument("binStart >= binEnd");
        if (stepBin <= 0)
            throw std::invalid_argument("binStep <= 0");
        if (coord < 0 || coord >= X.size())
            throw std::invalid_argument("coord out of range X");
        if (paramNumberA < 0 || paramNumberA >= params.size())
            throw std::invalid_argument("paramNumber out of range params param 1");
        if (paramNumberB < 0 || paramNumberB >= params.size())
            throw std::invalid_argument("paramNumber out of range params param 2");
        if (paramNumberB == paramNumberA)
            throw std::invalid_argument("param 1 == param 2");
        //    if (linspaceStartB == linspaceEndB) throw std::invalid_argument("linspaceStartB == linspaceEndB");
        //    if (linspaceStartA == linspaceEndA) throw std::invalid_argument("linspaceStartA == linspaceEndA");
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return std::vector<std::vector<double>>(0);
    }

    //Определение устройства, на котором будет запускаться программа
    int device = 0;
    cudaDeviceProp deviceProp;
    int numBlocks;
    int threadsPerBlock;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, device));
    // !Из-за большого колличества потоков в блоке, может появляться ошибка о нехватке памяти!
    // Для ее исправления стоит уменьшить максимальное колличество блоков
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock; 

    //Деление на итерации происходит в зависимости от свободной памяти на видеокарте freeMem(итерация - 1 запуск ядра GPU)
    size_t freeMem, totalMem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
    freeMem *= 0.8 / 1024 / 1024; // bytes

    //Выделение памяти для переменных и перенос их в константную память на GPU
    int XSize = X.size();
    int paramsSize = params.size();
    int histEntropySizeRow = linspaceNumA;
    int histEntropySizeCol = linspaceNumB;
    int histEntropySize = histEntropySizeRow * histEntropySizeCol;
    int binSize = static_cast<int>(std::ceil((endBin - startBin) / stepBin));
    std::vector<double> paramLinspaceA = linspaceNum(linspaceStartA, linspaceEndA, linspaceNumA);
    std::vector<double> paramLinspaceB = linspaceNum(linspaceStartB, linspaceEndB, linspaceNumB);

    double *d_X, *d_params, *d_paramLinspaceA;

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_startBin, &startBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_endBin, &endBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_stepBin, &stepBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tMax, &tMax, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_transTime, &transTime, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_h, &h, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_coord, &coord, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_XSize, &XSize, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramsSize, &paramsSize, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramNumberA, &paramNumberA, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramNumberB, &paramNumberB, sizeof(int)));

    // Выделение памяти для массивов, что будут передаваться в ядро
    CHECK_CUDA_ERROR(cudaMalloc(&d_X, XSize * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_params, paramsSize * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_X, X.data(), XSize * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_params, params.data(), paramsSize * sizeof(double), cudaMemcpyHostToDevice));

    // Определение необходимого колличества итераций
    // 1. Считается общее необходимое колличество итераций
    // 2. Оценивается возможность выполнения этого в памяти видеокарты
    // 3. В случае, если памяти на GPU не хватает, происходит деление по B на две половины. (Если b_size == 1, то происходит деление по A)
    double bytes = static_cast<double>(histEntropySize) * sizeof(double) / (1024 * 1024) + static_cast<double>(histEntropySize) * static_cast<double>(binSize) * sizeof(double) / (1024 * 1024);

    int iteratationsB = 1;
    int iteratationsA = 1;

    double memEnabledB = linspaceNumB;
    double memEnabledA = linspaceNumA;

    while ((((memEnabledA * std::ceil(memEnabledB) + static_cast<double>(binSize) * memEnabledA * std::ceil(memEnabledB)) * sizeof(double))) / (1024 * 1024) > freeMem)
    {
        if (std::ceil(memEnabledB) <= 1)
        {
            memEnabledA /= 2;
            iteratationsA *= 2;
        }
        else
        {
            memEnabledB /= 2;
            iteratationsB *= 2;
        }
    }

    memEnabledA = std::ceil(memEnabledA);
    memEnabledB = std::ceil(memEnabledB);

    std::cout << "freeMem: " << freeMem << "MB Needed Mbytes: " << bytes << "MB\n";
    std::cout << "iterationsB: " << iteratationsB << " B_size: " << " " << memEnabledB << "\n";
    std::cout << "iterationsA: " << iteratationsA << " A_size: " << " " << memEnabledA << "\n";

    std::vector<std::vector<double>> histEntropy2DFinal;
    histEntropy2DFinal.reserve(histEntropySizeCol);
    int SizeCol;
    double remainingB;
    double currentStepB;
    double processedB = 0;

    //Начало выполнения итераций. 
    for (int i = 0; i < iteratationsB; ++i)
    {
        // Массив B выделяется в соответсвии со свободной памятью GPU
        remainingB = histEntropySizeCol - processedB;
        currentStepB = std::min(memEnabledB, remainingB);

        std::vector<double> partParamLinspaceB(currentStepB);

        std::copy(
            paramLinspaceB.begin() + processedB,
            paramLinspaceB.begin() + processedB + currentStepB,
            partParamLinspaceB.begin());

        processedB += currentStepB;
        SizeCol = partParamLinspaceB.size();

        double *d_paramLinspaceB, *d_histEntropy;

        int SizeRow;
        double remainingA;
        double currentStepA;
        double processedA = 0;

        std::vector<double> histEntropyRowFinal;

        for (int j = 0; j < iteratationsA; ++j)
        {
            // Массив A выделяется в соответсвии со свободной памятью GPU
            remainingA = histEntropySizeRow - processedA;
            currentStepA = std::min(memEnabledA, remainingA);

            std::vector<double> partParamLinspaceA(currentStepA);

            std::copy(
                paramLinspaceA.begin() + processedA,
                paramLinspaceA.begin() + processedA + currentStepA,
                partParamLinspaceA.begin());

            processedA += currentStepA;
            SizeRow = partParamLinspaceA.size();

            histEntropySize = SizeRow * SizeCol;
            std::vector<double> histEntropy(histEntropySize);

            //Когда границы итерации установлены, выделяется память под все переменные, зависящее от размера A и B
            CHECK_CUDA_ERROR(cudaMalloc(&d_histEntropy, histEntropySize * sizeof(double)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_paramLinspaceB, SizeCol * sizeof(double)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_paramLinspaceA, SizeRow * sizeof(double)));

            CHECK_CUDA_ERROR(cudaMemcpy(d_paramLinspaceB, partParamLinspaceB.data(), SizeCol * sizeof(double), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_paramLinspaceA, partParamLinspaceA.data(), SizeRow * sizeof(double), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySize, &histEntropySize, sizeof(int)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySizeCol, &SizeCol, sizeof(int)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySizeRow, &SizeRow, sizeof(int)));

            double *bins;
            cudaMalloc(&bins, (histEntropySize * binSize) * sizeof(double));
            cudaMemset(bins, 0, histEntropySize * binSize * sizeof(double));

            // Определение колличества потоков и блоков для запуска ядра
            // Если данных на блок не очень много, то программа старается выделить блоков в 4 раза больше чем колличество мультипроцессоров
            // Данный подход показывает лучшую производительность 
            numBlocks = deviceProp.multiProcessorCount * 4;
            threadsPerBlock = std::ceil(histEntropySize / (float)numBlocks);

            if (threadsPerBlock > maxThreadsPerBlock)
            {
                threadsPerBlock = maxThreadsPerBlock;
                numBlocks = std::ceil(histEntropySize / (float)threadsPerBlock);
            }
            else if (threadsPerBlock == 0)
                threadsPerBlock = 1;
            std::cout << "Memory block is: " << i << " / " << iteratationsB << "\n";
            std::cout << "blocks: " << numBlocks << " threads: " << threadsPerBlock << " sm's: " << deviceProp.multiProcessorCount << "\n";
            int progress = 0;
            cudaMemcpyToSymbol(d_progress, &progress, sizeof(int));

            // Запуск функции ядра
            calculateHistEntropyCuda3D<<<numBlocks, threadsPerBlock>>>(
                d_X, d_params, d_paramLinspaceA, d_paramLinspaceB, d_histEntropy, bins);

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
                throw std::runtime_error("CUDA kernel execution failed");
            }

            // Запись финального массива в память
            std::vector<std::vector<double>> histEntropy2D(SizeCol, std::vector<double>(SizeRow));
            cudaMemcpy(histEntropy.data(), d_histEntropy, histEntropySize * sizeof(double), cudaMemcpyDeviceToHost);

            if (memEnabledB == 1)
            {
                histEntropyRowFinal.insert(histEntropyRowFinal.end(), histEntropy.begin(), histEntropy.end());
            }
            else
            {
                for (int q = 0; q < SizeRow; ++q)
                {
                    for (int s = 0; s < SizeCol; ++s)
                    {
                        histEntropy2D[s][q] = std::move(histEntropy[q * SizeCol + s]);
                    }
                }
                for (int q = 0; q < SizeCol; ++q)
                {
                    histEntropy2DFinal.push_back(histEntropy2D[q]);
                }
            }

            cudaFree(bins);
            cudaFree(d_paramLinspaceA);
        }

        if (memEnabledB == 1)
        {
            histEntropy2DFinal.push_back(histEntropyRowFinal);
        }
        cudaFree(d_paramLinspaceB);
    }

    //--- Освобождение памяти ---
    cudaFree(d_X);
    cudaFree(d_params);

    return histEntropy2DFinal;
}

/**
 * @brief Функция для вычисления энтропии по 1 параметру с использованием CUDA.
 * 
 * 
 * @param transTime Время транзиенты.
 * @param tMax Время работы системы.
 * @param h Шаг.
 * @param X координаты системы.
 * @param coord Индекс координаты по которой будет строится энтропия.
 * @param params Вектор параметров, где params[0] - коэффициент симметрии.
 * @param paramNumberA Индекс параметра для вычислений в params.
 * @param startBin Начало диапазона для столбцов гистограмм.
 * @param endBin Конец диапазона для столбцов гистограмм.
 * @param stepBin Шаг столбцов гистограмм.
 * @param linspaceStartA Начало диапазона для массива параметра.
 * @param linspaceEndA Конец диапазона для массива параметра.
 * @param linspaceNumA Количество точек в массиве параметра.
 * 
 * @return std::vector<double> Возвращает энтропию по 1 параметру, вычисленную для данных.
 */
__host__ std::vector<double> histEntropyCUDA2D(
    const double transTime, const double tMax, const double h,
    const std::vector<double> &X, const int coord,
    std::vector<double> &params, const int paramNumberA,
    const double startBin, const double endBin, const double stepBin,
    double linspaceStartA, double linspaceEndA, int linspaceNumA)
{
    // В конец парамтров добавляется "пустышка" чтобы использовать ее как второй параметр и запускать функцию по вычислению энтропии по 2 параметрам
    params.push_back(0);
    int paramNumberB = params.size() - 1;
    double linspaceStartB = params[paramNumberB];
    double linspaceEndB = linspaceStartB;
    double linspaceNumB = 1;

    std::vector<std::vector<double>> histEntropy3D = histEntropyCUDA3D(transTime, tMax, h, X, coord, params, paramNumberA, paramNumberB,
                                                                       startBin, endBin, stepBin, linspaceStartA, linspaceEndA, linspaceNumA,
                                                                       linspaceStartB, linspaceEndB, linspaceNumB);

    return histEntropy3D[0];
}


/*
Небольшое резюме:

При большом размере диапазона гистограммы (startbins, endbins, stepbins) памяти на поток в GPU может не хватить для хранения всех значений.
В связи с этим мы приняли решение сохранять информацию о них в глобальной памяти.
Для этого в глобальной памяти был создан массив, который хранит все значения bins для всех потоков.
Этот массив получается очень большим, но иначе у программы не будет возможности работать с большими диапазонами гистограммы.

Также было добавлено разбиение запуска ядра на итерации в зависимости от размеров массивов параметров.
Если параметров слишком много и массив bins не помещается в память программы,
то параметры будут разбиты на несколько итераций для последующего отдельного запуска.

Мы проверили работу программы на двух системах: системе Ресслера и системе, которую использовали на лабораторных.
В обоих случаях результат соответствовал правильным данным.

Скорость работы программы была оптимизирована насколько возможно. Проверка в Nsight Compute показала,
что по всем параметрам, кроме количества warp'ов, программа работает без возможности существенного ускорения (для наших ПК).

Nsight указал, что теоретически возможно ускорение на 33.3% из-за использования 8 блоков из 12 в warp'ах.
При этом инструмент сообщил, что не может запускать 12/12 warp'ов из-за большого количества регистров.
Однако, насколько мы поняли, в данной программе мы не можем уменьшить их количество из-за большого глобального массива bins для хранения гистограмм.

Сообщение, которое выдает Nsight:
The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12.
This kernel's theoretical occupancy (66.7%) is limited by the number of required registers.

*/