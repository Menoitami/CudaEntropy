#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>
#include "lib.cuh"


__global__ void calculateHistEntropyCuda3D(const double* const X,const int* XSize, const double* const params,
                                            const int* paramsSize, const int* paramNumberA, const int* paramNumberB, const double* const paramLinspaceA,
                                            const double* const paramLinspaceB, const int* histEntropySizeRow,  const int* histEntropySizeCol,
                                            int* coord, double* tMax, double* transTime, double* h, double* startBin, double* endBin,
                                            double* stepBin, double* histEntropy, int* histEntropySize , double** bins_global) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (idx < *histEntropySize) {

        int row = idx/ *histEntropySizeCol;
        int col = idx % *histEntropySizeCol;

        double* params_local = new double[*paramsSize];
        memcpy(params_local, params, *paramsSize * sizeof(double));   

        params_local[*paramNumberA] = paramLinspaceA[row]; 
        params_local[*paramNumberB] = paramLinspaceB[col]; 

        double* X_local = new double[*XSize];
        memcpy(X_local, X, *XSize * sizeof(double));

        loopCalculateDiscreteModel(X_local, params_local, *h, static_cast<int>(*transTime / *h), 0, 0, 0,nullptr,0, 0);
        int binSize = static_cast<int>(ceil((*endBin - *startBin) / *stepBin));
        int sum = 0;

        CalculateHistogram(
             *tMax, *h, X_local, params_local, *coord, *startBin, *endBin, *stepBin, sum, bins_global[idx]);

        double H = calculateEntropy(bins_global[idx], binSize, sum);
        histEntropy[row * *histEntropySizeCol + col] = (H / __log2f(binSize));

        delete[] X_local; 
        delete[] params_local;
    }
}


__host__ std::vector<double> histEntropyCUDA3D(
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
        return std::vector<double>(0); 
    }

    int XSize = X.size();
    int paramsSize = params.size();
    int histEntropySizeRow = paramLinspaceA.size();
    int histEntropySizeCol = paramLinspaceB.size();

    int histEntropySize = histEntropySizeRow * histEntropySizeCol;

    std::vector<double> histEntropy(histEntropySize);

    // Выделяем память для bins как двумерный массив для всех потоков
    int binSize = static_cast<int>(ceil((endBin - startBin) / stepBin));

    // --- Выделение памяти на устройстве ---
    double *d_X, *d_params, *d_paramLinspaceA,*d_paramLinspaceB, *d_histEntropy;
    int *d_coord, *d_XSize, *d_paramsSize, *d_paramNumberA, *d_paramNumberB, *d_histEntropySize,*d_histEntropySizeRow,*d_histEntropySizeCol;
    double *d_tMax, *d_transTime, *d_h, *d_startBin, *d_endBin, *d_stepBin;
    
    // глобальная памят все
    cudaMalloc((void**)&d_X, XSize * sizeof(double));
    cudaMalloc((void**)&d_params, paramsSize * sizeof(double));
    cudaMalloc((void**)&d_paramLinspaceA, histEntropySizeRow * sizeof(double));
    cudaMalloc((void**)&d_paramLinspaceB, histEntropySizeCol * sizeof(double));
    cudaMalloc((void**)&d_histEntropy, histEntropySize * sizeof(double));
    cudaMalloc((void**)&d_histEntropySize, sizeof(int));
    cudaMalloc((void**)&d_histEntropySizeRow, sizeof(int));
    cudaMalloc((void**)&d_histEntropySizeCol, sizeof(int));
    cudaMalloc((void**)&d_XSize, sizeof(int));
    cudaMalloc((void**)&d_paramsSize, sizeof(int));
    cudaMalloc((void**)&d_paramNumberA, sizeof(int));
    cudaMalloc((void**)&d_paramNumberB, sizeof(int));
    cudaMalloc((void**)&d_coord, sizeof(int));
    cudaMalloc((void**)&d_tMax, sizeof(double));
    cudaMalloc((void**)&d_transTime, sizeof(double));
    cudaMalloc((void**)&d_h, sizeof(double));
    cudaMalloc((void**)&d_startBin, sizeof(double));
    cudaMalloc((void**)&d_endBin, sizeof(double));
    cudaMalloc((void**)&d_stepBin, sizeof(double));


    // --- Копирование данных с хоста на устройство ---
    cudaMemcpy(d_X, X.data(), XSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, params.data(), paramsSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_paramLinspaceA, paramLinspaceA.data(), histEntropySizeRow * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_paramLinspaceB, paramLinspaceB.data(), histEntropySizeCol * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histEntropy, histEntropy.data(), histEntropySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histEntropySize, &histEntropySize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histEntropySizeRow, &histEntropySizeRow, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histEntropySizeCol, &histEntropySizeCol, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_XSize, &XSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_paramsSize, &paramsSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_paramNumberA, &paramNumberA, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_paramNumberB, &paramNumberB, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coord, &coord, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tMax, &tMax, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transTime, &transTime, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, &h, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_startBin, &startBin, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_endBin, &endBin, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stepBin, &stepBin, sizeof(double), cudaMemcpyHostToDevice);

    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    
    int sharedMemPerBlock = deviceProp.sharedMemPerBlock;

    int threadsPerBlock = deviceProp.maxThreadsPerBlock;

    int numBlocks = std::ceil((histEntropySize + threadsPerBlock - 1) / threadsPerBlock);

    std::cout<<sharedMemPerBlock<<" "<<numBlocks<<" "<<threadsPerBlock<<"\n";
    
    double** hostBins = (double**)malloc(histEntropySize * sizeof(double*));
    for (int i = 0; i < histEntropySize; ++i) {
        cudaMalloc(&hostBins[i], binSize * sizeof(double)); // выделяем для binSize
    }

    double** bins;
    cudaMalloc(&bins, histEntropySize * sizeof(double*)); // только histEntropySize указателей
    cudaMemcpy(bins, hostBins, histEntropySize * sizeof(double*), cudaMemcpyHostToDevice);
   
    

    

    calculateHistEntropyCuda3D<<<numBlocks, threadsPerBlock>>>(d_X, d_XSize, d_params, d_paramsSize,
                                            d_paramNumberA,d_paramNumberB, d_paramLinspaceA,d_paramLinspaceB,d_histEntropySizeRow,d_histEntropySizeCol, d_coord, d_tMax, d_transTime, d_h,
                                            d_startBin, d_endBin, d_stepBin, d_histEntropy, d_histEntropySize,bins);

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
    cudaMemcpy(histEntropy.data(), d_histEntropy, histEntropySize * sizeof(double), cudaMemcpyDeviceToHost);

    //--- Освобождение памяти ---
    cudaFree(d_X);
    cudaFree(d_params);
    cudaFree(d_paramLinspaceA);
    cudaFree(d_paramLinspaceB);
    cudaFree(d_histEntropy);
    cudaFree(d_XSize);
    cudaFree(d_paramsSize);
    cudaFree(d_paramNumberA);
    cudaFree(d_paramNumberB);
    cudaFree(d_coord);
    cudaFree(d_tMax);
    cudaFree(d_transTime);
    cudaFree(d_h);
    cudaFree(d_startBin);
    cudaFree(d_endBin);
    cudaFree(d_stepBin);
    cudaFree(d_histEntropySize);
    cudaFree(d_histEntropySizeRow);
    cudaFree(d_histEntropySizeCol);

    for (int i = 0; i < histEntropySize; ++i) {
        cudaFree(hostBins[i]);
    }
    cudaFree(bins);
    free(hostBins);

    return histEntropy;
}



int main() {
   
    // Задаем параметры для расчета
    double transTime = 1000;  // Время переходного процесса
    double tMax = 2000;       // Время моделирования после TT
    double h = 0.01;          // Шаг интегрирования

    std::vector<double> X = {0.1, 0.1, 0}; // Начальное состояние
    int coord = 0;                        // Координата для анализа

    std::vector<double> params = {0, 0.2, 0.2, 5.7}; // Параметры модели
    

    double startBin = -20; // Начало гистограммы
    double endBin = 20;    // Конец гистограммы
    double stepBin = 0.1;  // Шаг бинов гистограммы

    // Параметры для linspace
    double linspaceStartA = 0.1;  // Начало диапазона параметра
    double linspaceEndA = 0.35;   // Конец диапазона параметра
    int linspaceNumA = 400;       // Количество точек параметра
    std::vector<double> paramLinspaceA = linspace(linspaceStartA, linspaceEndA, linspaceNumA);
    int paramNumberA = 1;         // Индекс параметра для анализа


    double linspaceStartB = 0.1;  // Начало диапазона параметра
    double linspaceEndB = 0.35;   // Конец диапазона параметра
    int linspaceNumB = 400;       // Количество точек параметра
    std::vector<double> paramLinspaceB = linspace(linspaceStartB, linspaceEndB, linspaceNumB);
    int paramNumberB = 2;         // Индекс параметра для анализа
    //Вызов функции histEntropyCUDA2D
    std::vector<double> histEntropy = histEntropyCUDA3D(
                                        transTime, tMax, h,
                                        X, coord,
                                        params, paramNumberA,paramNumberB,
                                        startBin, endBin, stepBin,
                                        paramLinspaceA, paramLinspaceB
                                    );

    

    // histEntropyCUDA2D(
    //     1000, 2000, 0.01,
    //     {0.1, 0.1, 0}, 0,
    //     {0, 0.2, 0.2, 5.7}, 1,
    //     -20, 20, 0.1,
    //     0.1, 0.35, 400
    // );

    return 0;
}