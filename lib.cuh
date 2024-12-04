#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <thread>

#define CHECK_CUDA_ERROR(call)                                             \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)         \
                      << " in file " << __FILE__ << " at line " << __LINE__ \
                      << std::endl;                                        \
            exit(err);                                                     \
        }                                                                  \
    }


__constant__ int d_histEntropySize;
__constant__ int d_histEntropySizeRow;
__constant__ int d_histEntropySizeCol;
__constant__ double d_startBin;
__constant__ double d_endBin;
__constant__ double d_stepBin;
__constant__ double d_tMax;
__constant__ double d_transTime;
__constant__ double d_h;
__constant__ int d_coord;
__constant__ int d_paramsSize;
__constant__ int d_XSize;
__constant__ int d_paramNumberA;
__constant__ int d_paramNumberB;

__device__ int d_progress;
__device__ const double EPSD = std::numeric_limits<double>::epsilon();


__host__ std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result;

    if (num<0) throw std::invalid_argument( "received negative number of points" );
    if (num == 0) return result;
    if (num == 1) {
        result.push_back(start); 
        return result;
    }

    double step = (end - start) / (num - 1); 

    for (int i = 0; i < num; ++i) {
        result.push_back(start + i * step);
    }

    return result;
}

__host__ void writeToCSV(const std::vector<std::vector<double>>& histEntropy2D, int cols, int rows, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outFile << histEntropy2D[i][j];
            if (j < cols - 1) {
                outFile << ","; 
            }
        }
        outFile << "\n"; 
    }

    outFile.close();  
    std::cout << "Data successfully written to file " << filename << std::endl;
}
std::vector<std::vector<double>> convert1DTo2D(const std::vector<double>& histEntropy1D) {
    return {histEntropy1D};
}
__host__ void writeToCSV(const std::vector<double>& histEntropy1D, int cols, const std::string& filename) {
    auto histEntropy2D = convert1DTo2D(histEntropy1D);
    writeToCSV(histEntropy2D, cols, 1, filename);
}

__device__ __host__ void calculateDiscreteModel(double* x, const double* a, const double h)
{
	/**
	 * here we abstract from the concept of parameter names. 
	 * ALL parameters are numbered with indices. 
	 * In the current example, the parameters go like this:
	 * 
	 * values[0] - sym
	 * values[1] - A
	 * values[2] - B
	 * values[3] - C
	 */

	//x[0] = x[0] + h * (-x[1] - x[2]);
	//x[1] = x[1] + h * (x[0] + a[0] * x[1]);
	//x[2] = x[2] + h * (a[1] + x[2] * (x[0] - a[2]));

    double h1 = 0.5 * h + a[0];
    double h2 = 0.5 * h - a[0];

    
    x[0] += h1 * (-x[1] - x[2]);
    x[1] += h1 * (x[0] + a[1] * x[1]);
    x[2] += h1 * (a[2] + x[2] * (x[0] - a[3]));

    x[2] = (x[2] + h2 * a[2]) / (1 - h2 * (x[0] - a[3]));
    x[1] = (x[1] + h2 * x[0]) / (1 - h2 * a[1]);
    x[0] += h2 * (-x[1] - x[2]);

}

__device__ __host__ bool loopCalculateDiscreteModel(double* x, const double* values, const double h,
 const int amountOfIterations, const int preScaller,
 int writableVar, const double maxValue, double* data, const int startDataIndex, const int writeStep)
{
 for (int i = 0; i < amountOfIterations; ++i)
 {
  if (data != nullptr)
   data[startDataIndex + i * writeStep] = x[writableVar];

  for (int j = 0; j < preScaller - 1; ++j) calculateDiscreteModel(x, values, h);

  calculateDiscreteModel(x, values, h);

  if (maxValue != 0)
   if (fabsf(x[writableVar]) > maxValue)
    return false;
 }
 return true;
}

__device__  double calculateEntropy(double* bins, int binSize, const int sum) {
    double entropy = 0.0;

    for (int i = 0; i < binSize; ++i) {
        if (bins[i] > 0) {
            bins[i] = (bins[i] / (sum + EPSD));
            entropy -= (bins[i] - EPSD) * log2(bins[i]); 
        }
    }

    return entropy;
}


__device__ __host__ void CalculateHistogram(
    double* X, const double* param,
     int& sum, double* bins) {


    int iterations = static_cast<int>(d_tMax / d_h);
    double last = X[d_coord];
    bool lastBigger = false;

    int binSize = static_cast<int>(ceil((d_endBin - d_startBin) / d_stepBin));

    for (int i = 0; i < binSize; ++i) {
        bins[i] = 0.0;
    }


    for (int i = 0; i < iterations; ++i) {
        calculateDiscreteModel(X, param, d_h);

        if (X[d_coord] > last) {
            lastBigger = true;
        } else if (X[d_coord] < last && lastBigger) {
            if (last >= d_startBin && last < d_endBin) {
                int index = static_cast<int>((last - d_startBin) / d_stepBin);
                bins[index]++;
                sum++;
            }
            lastBigger = false;
        }
        last = X[d_coord];
    }

}

/*
__global__ void calculateHistEntropyCuda2D(const double* const X,const int* XSize, const double* const params,
                                            const int* paramsSize, const int* paramNumber,const double* const paramLinspace,
                                            int* coord, double* tMax, double* transTime, double* h, double* startBin,
                                            double* endBin, double* stepBin, double* histEntropy, int* histEntropySize , double** bins_global) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < *histEntropySize) {

        double* params_local = new double[*paramsSize];
        memcpy(params_local, params, *paramsSize * sizeof(double));   

        params_local[*paramNumber] = paramLinspace[idx];

        double* X_local = new double[*XSize]; 
        memcpy(X_local, X, *XSize * sizeof(double));

        loopCalculateDiscreteModel(X_local, params_local, *h, static_cast<int>(*transTime / *h), 0, 0, 0,nullptr,0, 0);
        int binSize = static_cast<int>(ceil((*endBin - *startBin) / *stepBin));
        int sum = 0;

        CalculateHistogram(
             *tMax, *h, X_local, params_local, *coord, *startBin, *endBin, *stepBin, sum, bins_global[idx]);

        double H = calculateEntropy(bins_global[idx], binSize, sum);
        histEntropy[idx] = (H / __log2f(binSize));

        delete[] X_local; 
        delete[] params_local;
    }
}
*/
__global__ void calculateHistEntropyCuda3D(const double* const X, 
                                           const double* const params,
                                           const double* const paramLinspaceA,
                                           const double* const paramLinspaceB,
                                           double* histEntropy, 
                                           double** bins_global) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    //printf("Accessing bins_global at idx: %d\n",d_histEntropySize);
    if (idx < d_histEntropySize) {

        int row = idx/ d_histEntropySizeCol;
        int col = idx % d_histEntropySizeCol;

        double* params_local = new double[d_paramsSize];
        double* X_local = new double[d_XSize];

        memcpy(params_local, params, d_paramsSize * sizeof(double));
        memcpy(X_local, X, d_XSize * sizeof(double));

        params_local[d_paramNumberA] = paramLinspaceA[row];
        params_local[d_paramNumberB] = paramLinspaceB[col];

        loopCalculateDiscreteModel(X_local, params_local, d_h, 
                                   static_cast<int>(d_transTime / d_h), 
                                   0, 0, 0, nullptr, 0, 0);

        int binSize = static_cast<int>(ceil((d_endBin - d_startBin) / d_stepBin));
        int sum = 0;

        CalculateHistogram(
             X_local, params_local, sum, bins_global[idx]);

        double H = calculateEntropy(bins_global[idx], binSize, sum);

        // Нормализуем и сохраняем результат в глобальную память
        histEntropy[row * d_histEntropySizeCol + col] = (H / __log2f(binSize));

        delete[] params_local;
        delete[] X_local;

        // Вывод прогресса
        
        int progress = atomicAdd(&d_progress, 1);
        if ((progress + 1) % (d_histEntropySize / 10) == 0) {
            printf("Progress: %d%%\n", ((progress + 1) * 100) / d_histEntropySize);
        }


    }
}

/*   Рассчет гистограмной энтропии по одному параметру
     transTime  - Время переходного процесса
     tMax -   Время моделирования после TT
     h = 0.01;          - Шаг интегрирования

     vector<> X - Начальное состояние
     coord = 0;                        - Координата для анализа

     vector<> params - набор параметров модели
     paramNumber - Индекс параметра для анализа

     startBin  - Начало гистограммы
     endBin  - Конец гистограммы
     stepBin - Шаг бинов гистограммы

     Параметры для linspace
     linspaceStart  - Начало диапазона параметра
     linspaceEnd  - Конец диапазона параметра
     linspaceNum  - Количество точек параметра*/
     /*
__host__ std::vector<double> histEntropyCUDA2D(
    const double transTime,const double tMax,const double h, 
    const std::vector<double>& X,const int coord, 
    const std::vector<double>& params,const int paramNumber, 
    const double startBin,const double endBin, const double stepBin, 
    const std::vector<double>& paramLinspace
)
 {
    // Проверки переменных
    try{
    if (tMax <= 0) throw std::invalid_argument("tMax <= 0");
    if (transTime <= 0) throw std::invalid_argument("transTime <= 0");
    if (h <= 0) throw std::invalid_argument("h <= 0");
    if (startBin >= endBin) throw std::invalid_argument("binStart >= binEnd");
    if (stepBin <= 0) throw std::invalid_argument("binStep <= 0");
    if (coord < 0 || coord >= X.size()) throw std::invalid_argument("coord out of range X");
    if (paramNumber < 0 || paramNumber >= params.size()) throw std::invalid_argument("paramNumber out of range params");
    }
     catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return std::vector<double>(0); 
    }

    int XSize = X.size();
    int paramsSize = params.size();
    int histEntropySize = paramLinspace.size();

    std::vector<double> HistEntropy(histEntropySize);

    int binSize = static_cast<int>(ceil((endBin - startBin) / stepBin));


    // --- Выделение памяти на устройстве ---
    double *d_X, *d_params, *d_paramLinspace, *d_histEntropy;
    int *d_coord, *d_XSize, *d_paramsSize, *d_paramNumber, *d_histEntropySize;
    double *d_tMax, *d_transTime, *d_h, *d_startBin, *d_endBin, *d_stepEdge;
    
    // глобальная памят все
    cudaMalloc((void**)&d_X, XSize * sizeof(double));
    cudaMalloc((void**)&d_params, paramsSize * sizeof(double));
    cudaMalloc((void**)&d_paramLinspace, histEntropySize * sizeof(double));
    cudaMalloc((void**)&d_histEntropy, histEntropySize * sizeof(double));
    cudaMalloc((void**)&d_XSize, sizeof(int));
    cudaMalloc((void**)&d_paramsSize, sizeof(int));
    cudaMalloc((void**)&d_paramNumber, sizeof(int));
    cudaMalloc((void**)&d_coord, sizeof(int));
    cudaMalloc((void**)&d_tMax, sizeof(double));
    cudaMalloc((void**)&d_transTime, sizeof(double));
    cudaMalloc((void**)&d_h, sizeof(double));
    cudaMalloc((void**)&d_startBin, sizeof(double));
    cudaMalloc((void**)&d_endBin, sizeof(double));
    cudaMalloc((void**)&d_stepEdge, sizeof(double));
    cudaMalloc((void**)&d_histEntropySize, sizeof(int));
    

    // --- Копирование данных с хоста на устройство ---
    cudaMemcpy(d_X, X.data(), XSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, params.data(), paramsSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_paramLinspace, paramLinspace.data(), histEntropySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histEntropy, HistEntropy.data(), histEntropySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_XSize, &XSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_paramsSize, &paramsSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_paramNumber, &paramNumber, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coord, &coord, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tMax, &tMax, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transTime, &transTime, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, &h, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_startBin, &startBin, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_endBin, &endBin, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stepEdge, &stepBin, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histEntropySize, &histEntropySize, sizeof(int), cudaMemcpyHostToDevice);
    
    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    int threadsPerBlock = deviceProp.maxThreadsPerBlock;

    int numBlocks = std::ceil((histEntropySize + threadsPerBlock - 1) / threadsPerBlock);

    std::cout<<"Blocks: "<<numBlocks<<" threads: "<<threadsPerBlock<<"\n";
    
    double** hostBins = (double**)malloc(histEntropySize * sizeof(double*));
    for (int i = 0; i < histEntropySize; ++i) {
        cudaMalloc(&hostBins[i], binSize * sizeof(double)); // выделяем для binSize
    }

    double** bins;
    cudaMalloc(&bins, histEntropySize * sizeof(double*)); // только histEntropySize указателей
    cudaMemcpy(bins, hostBins, histEntropySize * sizeof(double*), cudaMemcpyHostToDevice);
   

    calculateHistEntropyCuda2D<<<numBlocks, threadsPerBlock>>>(d_X, d_XSize, d_params, d_paramsSize,
                                            d_paramNumber, d_paramLinspace, d_coord, d_tMax, d_transTime, d_h,
                                            d_startBin, d_endBin, d_stepEdge, d_histEntropy, d_histEntropySize,bins);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA kernel execution failed");
    }
    

    cudaDeviceSynchronize();
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "Free memory: " << freeMem / (1024 * 1024) << " MB\n";
    std::cout << "Total memory: " << totalMem / (1024 * 1024) << " MB\n";

    // --- Копирование результата обратно на хост ---
    cudaMemcpy(HistEntropy.data(), d_histEntropy, histEntropySize * sizeof(double), cudaMemcpyDeviceToHost);


    // --- Освобождение памяти ---
    cudaFree(d_X);
    cudaFree(d_params);
    cudaFree(d_paramLinspace);
    cudaFree(d_histEntropy);
    cudaFree(d_XSize);
    cudaFree(d_paramsSize);
    cudaFree(d_paramNumber);
    cudaFree(d_coord);
    cudaFree(d_tMax);
    cudaFree(d_transTime);
    cudaFree(d_h);
    cudaFree(d_startBin);
    cudaFree(d_endBin);
    cudaFree(d_stepEdge);
    cudaFree(d_histEntropySize);

    for (int i = 0; i < histEntropySize; ++i) {
        cudaFree(hostBins[i]);
    }
    cudaFree(bins);
    free(hostBins);

    return HistEntropy;
}
*/
/*   Рассчет гистограмной энтропии по одному параметру
     transTime  - Время переходного процесса
     tMax -   Время моделирования после TT
     h = 0.01;          - Шаг интегрирования

     vector<> X - Начальное состояние
     coord = 0;                        - Координата для анализа

     vector<> params - набор параметров модели
     paramNumber - Индекс параметра для анализа

     startBin  - Начало гистограммы
     endBin  - Конец гистограммы
     stepBin - Шаг бинов гистограммы

     Параметры для linspace
     linspaceStart  - Начало диапазона параметра
     linspaceEnd  - Конец диапазона параметра
     linspaceNum  - Количество точек параметра*/
     


__host__ std::vector<std::vector<double>> histEntropyCUDA3D(
    const double transTime,const double tMax,const double h, 
    const std::vector<double>& X,const int coord, 
    const std::vector<double>& params,const int paramNumberA,const int paramNumberB, 
    const double startBin, const double endBin,const double stepBin, 
    const std::vector<double>& paramLinspaceA,const std::vector<double>& paramLinspaceB
)
 {
    
    try {
        if (tMax <= 0) throw std::invalid_argument("tMax <= 0");
        if (transTime <= 0) throw std::invalid_argument("transTime <= 0");
        if (h <= 0) throw std::invalid_argument("h <= 0");
        if (startBin >= endBin) throw std::invalid_argument("binStart >= binEnd");
        if (stepBin <= 0) throw std::invalid_argument("binStep <= 0");
        if (coord < 0 || coord >= X.size()) throw std::invalid_argument("coord out of range X");
        if (paramNumberA < 0 || paramNumberA >= params.size()) throw std::invalid_argument("paramNumber out of range params param 1");
        if (paramNumberB < 0 || paramNumberB >= params.size()) throw std::invalid_argument("paramNumber out of range params param 2");
        if (paramNumberB == paramNumberA) throw std::invalid_argument("param 1 == param 2");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return std::vector<std::vector<double>>(0); 
    }

    int XSize = X.size();
    int paramsSize = params.size();
    int histEntropySizeRow = paramLinspaceA.size();
    int histEntropySizeCol = paramLinspaceB.size();
    int histEntropySize = histEntropySizeRow * histEntropySizeCol;
    int binSize = static_cast<int>(ceil((endBin - startBin) / stepBin));
    
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_startBin, &startBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_endBin, &endBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_stepBin, &stepBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tMax, &tMax, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_transTime, &transTime, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_h, &h, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_coord, &coord, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_XSize, &XSize, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramsSize, &paramsSize, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySize, &histEntropySize, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySizeRow, &histEntropySizeRow, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySizeCol, &histEntropySizeCol, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramNumberA, &paramNumberA, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramNumberB, &paramNumberB, sizeof(int)));

    std::cout<<histEntropySize<<"\n";
    std::vector<double> histEntropy(histEntropySize);

    // --- Выделение памяти на устройстве ---
    double *d_X, *d_params, *d_paramLinspaceA, *d_paramLinspaceB, *d_histEntropy;
    
    // глобальная памят все
    CHECK_CUDA_ERROR(cudaMalloc(&d_X, XSize * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_params, paramsSize * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_paramLinspaceA, histEntropySizeRow * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_paramLinspaceB, histEntropySizeCol * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_histEntropy, histEntropySize * sizeof(double)));


    CHECK_CUDA_ERROR(cudaMemcpy(d_X, X.data(), XSize * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_params, params.data(), paramsSize * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_paramLinspaceA, paramLinspaceA.data(), histEntropySizeRow * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_paramLinspaceB, paramLinspaceB.data(), histEntropySizeCol * sizeof(double), cudaMemcpyHostToDevice));

    int device = 0;
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, device));
    
    const int threadsPerBlock = deviceProp.maxThreadsPerBlock;
    const int numBlocks = std::ceil((histEntropySize + threadsPerBlock - 1) / threadsPerBlock);

    std::cout<<"blocks: "<<numBlocks<<" threads: "<<threadsPerBlock<<"\n";
    
    double** hostBins = (double**)malloc(histEntropySize * sizeof(double*));
    for (int i = 0; i < histEntropySize; ++i) {
        cudaMalloc(&hostBins[i], binSize * sizeof(double)); // выделяем для binSize
    }

    double** bins;
    cudaMalloc(&bins, histEntropySize * sizeof(double*)); // только histEntropySize указателей
    cudaMemcpy(bins, hostBins, histEntropySize * sizeof(double*), cudaMemcpyHostToDevice);

    int progress = 0;
    cudaMemcpyToSymbol(d_progress, &progress, sizeof(int));

    calculateHistEntropyCuda3D<<<numBlocks, threadsPerBlock>>>(
        d_X, d_params, d_paramLinspaceA, d_paramLinspaceB, d_histEntropy,bins);


    //Cинхронизация
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA kernel execution failed");
    }
    
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "Free memory: " << freeMem / (1024 * 1024) << " MB\n";
    std::cout << "Total memory: " << totalMem / (1024 * 1024) << " MB\n";

    // --- Копирование результата обратно на хост ---
    cudaMemcpy(histEntropy.data(), d_histEntropy, histEntropySize * sizeof(double), cudaMemcpyDeviceToHost);


    // Создаем 2D представление с использованием std::vector
    std::vector<std::vector<double>> histEntropy2D(histEntropySizeCol, std::vector<double>(histEntropySizeRow));

    for (int i = 0; i < histEntropySizeRow; ++i) {
        for (int j = 0; j < histEntropySizeCol; ++j) {
            // Перемещаем данные с учетом транспонирования
            histEntropy2D[j][i] = std::move(histEntropy[i * histEntropySizeCol + j]);
        }
    }

 
    //--- Освобождение памяти ---
    cudaFree(d_X);
    cudaFree(d_params);
    cudaFree(d_paramLinspaceA);
    cudaFree(d_paramLinspaceB);

    for (int i = 0; i < histEntropySize; ++i) {
        cudaFree(hostBins[i]);
    }
    cudaFree(bins);
    free(hostBins);

    return histEntropy2D;
}