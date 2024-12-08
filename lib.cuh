#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <thread>
#include <nvrtc.h>
#include <cooperative_groups.h>


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


__host__ std::vector<double> linspaceNum(double start, double end, int num) {
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

__host__ std::vector<double> linspaceStep(double start, double end, double step) {
    std::vector<double> result;

    if (step <= 0) throw std::invalid_argument("Step must be positive");
    if (start > end) throw std::invalid_argument("Start must be less than or equal to end");

    for (double value = start; value < end; value += step) {
        result.push_back(value);
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
    bool exceededMaxValue = (maxValue != 0);
    for (int i = 0; i < amountOfIterations; ++i)
    {
        if (data != nullptr)
        data[startDataIndex + i * writeStep] = x[writableVar];

        for (int j = 0; j < preScaller - 1; ++j) calculateDiscreteModel(x, values, h);

        calculateDiscreteModel(x, values, h);

        if (exceededMaxValue && fabsf(x[writableVar]) > maxValue)
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


__device__ void CalculateHistogram(
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


__global__ void calculateHistEntropyCuda3D(const double* const X, 
                                           const double* const params,
                                           const double* const paramLinspaceA,
                                           const double* const paramLinspaceB,
                                           double* histEntropy, 
                                           double** bins_global) {
const int x_size =3;
const int param_size = 4;

    __shared__ double X_sh[x_size];
    __shared__ double params_sh[param_size];
    if(threadIdx.x==0){
        memcpy(params_sh, params, param_size * sizeof(double));
        memcpy(X_sh, X, x_size * sizeof(double));
    }
    __syncthreads();
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < d_histEntropySize) {

        int row = idx/ d_histEntropySizeCol;
        int col = idx % d_histEntropySizeCol;

        double X_locals[x_size];
        double params_local[param_size];
        
        memcpy(params_local, params_sh, param_size * sizeof(double));
        memcpy(X_locals, X_sh, x_size * sizeof(double));

        params_local[d_paramNumberA] = paramLinspaceA[row];
        params_local[d_paramNumberB] = paramLinspaceB[col];

        loopCalculateDiscreteModel(X_locals, params_local, d_h, 
                                   static_cast<int>(d_transTime / d_h), 
                                   0, 0, 0, nullptr, 0, 0);

        int binSize = static_cast<int>(ceil((d_endBin - d_startBin) / d_stepBin));
        int sum = 0;

        CalculateHistogram(
             X_locals, params_local, sum, bins_global[idx]);

        double H = calculateEntropy(bins_global[idx], binSize, sum);

        // Нормализуем и сохраняем результат в глобальную память
        histEntropy[row * d_histEntropySizeCol + col] = (H / __log2f(binSize));


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

     Параметры для linspaceNum
     linspaceStart  - Начало диапазона параметра
     linspaceEnd  - Конец диапазона параметра
     linspaceNum  - Количество точек параметра*/
     


__host__ std::vector<std::vector<double>> histEntropyCUDA3D(
    const double transTime,const double tMax,const double h, 
    const std::vector<double>& X,const int coord, 
    const std::vector<double>& params,const int paramNumberA,const int paramNumberB, 
    const double startBin, const double endBin,const double stepBin, 
    double linspaceStartA, double linspaceEndA, int linspaceNumA,double linspaceStartB, double linspaceEndB, int linspaceNumB
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
        if (linspaceStartB == linspaceEndB) throw std::invalid_argument("linspaceStartB == linspaceEndB");
        if (linspaceStartA == linspaceEndA) throw std::invalid_argument("linspaceStartA == linspaceEndA");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return std::vector<std::vector<double>>(0); 
    }

    int device = 0;
    cudaDeviceProp deviceProp;
    int numBlocks;
    int threadsPerBlock;

    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, device));
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    int XSize = X.size();
    int paramsSize = params.size();
    int histEntropySizeRow = linspaceNumA;
    int histEntropySizeCol =  linspaceNumB;
    int histEntropySize = histEntropySizeRow * histEntropySizeCol;
    int binSize = static_cast<int>(std::ceil((endBin - startBin) / stepBin));
    std::vector<double> paramLinspaceA = linspaceNum(linspaceStartA, linspaceEndA,linspaceNumA);

    size_t freeMem, totalMem;

	CHECK_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
    freeMem*=0.7*1024*1024; //bytes

    long long int bytes =(histEntropySize+ binSize*histEntropySize)* sizeof(double); // bytes

    int iteratations = 1;
    float memEnabledB = linspaceNumB;


    while (((histEntropySizeRow*memEnabledB+ binSize*histEntropySizeRow*memEnabledB)* sizeof(double)) > freeMem ){

        memEnabledB /=2;
        iteratations*=2;

    }
    memEnabledB =std::ceil(memEnabledB);


    std::cout<<"freeMem: "<<freeMem/1024/1024<<"MB Needed bytes: "<<bytes/1024/1024<<"MB\n";
    std::cout<<"iterations: "<<iteratations<<" B_size: "<<" "<<memEnabledB<<"\n";

     // --- Выделение памяти на устройстве ---
    double *d_X, *d_params, *d_paramLinspaceA;

    CHECK_CUDA_ERROR(cudaMalloc(&d_X, XSize * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_params, paramsSize * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_paramLinspaceA, histEntropySizeRow * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_X, X.data(), XSize * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_params, params.data(), paramsSize * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_paramLinspaceA, paramLinspaceA.data(), histEntropySizeRow * sizeof(double), cudaMemcpyHostToDevice));


    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_startBin, &startBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_endBin, &endBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_stepBin, &stepBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tMax, &tMax, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_transTime, &transTime, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_h, &h, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_coord, &coord, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_XSize, &XSize, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramsSize, &paramsSize, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySizeRow, &histEntropySizeRow, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramNumberA, &paramNumberA, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramNumberB, &paramNumberB, sizeof(int)));


    std::vector<std::vector<double>> histEntropy2DFinal;
    histEntropy2DFinal.reserve(histEntropySizeCol);

    double stepB;
    if (linspaceNumB!=1)stepB = (linspaceEndB - linspaceStartB) / (static_cast<double>(linspaceNumB));
    else stepB = linspaceEndB - linspaceStartB+1;
    double startB = linspaceStartB - memEnabledB*stepB;
    double endB;

    for (int i = 0 ; i<iteratations;++i){

        startB = startB+ memEnabledB*stepB;
        endB = startB +memEnabledB*stepB<linspaceEndB ?startB +memEnabledB*stepB : linspaceEndB;

        std::vector<double> paramlinspaceB = linspaceStep(startB, endB, stepB);

        histEntropySizeCol = paramlinspaceB.size();
        histEntropySize = histEntropySizeRow * histEntropySizeCol;

        std::vector<double> histEntropy(histEntropySize);

        double *d_paramLinspaceB, *d_histEntropy;

        CHECK_CUDA_ERROR(cudaMalloc(&d_paramLinspaceB, histEntropySizeCol * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_histEntropy, histEntropySize * sizeof(double)));

        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySize, &histEntropySize, sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySizeCol, &histEntropySizeCol, sizeof(int)));

        CHECK_CUDA_ERROR(cudaMemcpy(d_paramLinspaceB, paramlinspaceB.data(), histEntropySizeCol * sizeof(double),cudaMemcpyHostToDevice));

        double** hostBins = (double**)malloc(histEntropySize * sizeof(double*));
        for (int i = 0; i < histEntropySize; ++i) {
            cudaMalloc(&hostBins[i], binSize * sizeof(double)); // выделяем для binSize
        }

        double** bins;
        cudaMalloc(&bins, histEntropySize * sizeof(double*)); // только histEntropySize указателей
        cudaMemcpy(bins, hostBins, histEntropySize * sizeof(double*), cudaMemcpyHostToDevice);


        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, device));

        numBlocks = deviceProp.multiProcessorCount * 4;
        threadsPerBlock = std::ceil(histEntropySize / (float)numBlocks);

        if (threadsPerBlock > maxThreadsPerBlock) {
            threadsPerBlock = maxThreadsPerBlock;
            numBlocks = std::ceil(histEntropySize / (float)threadsPerBlock);
        }   

        std::cout<<"blocks: "<<numBlocks<<" threads: "<<threadsPerBlock<<" sm's: " << deviceProp.multiProcessorCount<<"\n";

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

        cudaMemcpy(histEntropy.data(), d_histEntropy, histEntropySize * sizeof(double), cudaMemcpyDeviceToHost);

        std::vector<std::vector<double>> histEntropy2D(histEntropySizeCol, std::vector<double>(histEntropySizeRow));

        for (int i = 0; i < histEntropySizeRow; ++i) {
            for (int j = 0; j < histEntropySizeCol; ++j) {

                histEntropy2D[j][i] = std::move(histEntropy[i * histEntropySizeCol + j]);
            }
        }

        for (auto& row : histEntropy2D) {
        histEntropy2DFinal.push_back(std::move(row));
        }

        cudaFree(d_paramLinspaceB);
        for (int i = 0; i < histEntropySize; ++i) {
            cudaFree(hostBins[i]);
        }
        cudaFree(bins);
        free(hostBins);

    }


    // cudaMemGetInfo(&freeMem, &totalMem);
    // std::cout << "Free memory: " << freeMem / (1024 * 1024) << " MB\n";
    // std::cout << "Total memory: " << totalMem / (1024 * 1024) << " MB\n";
    // std::cout << "Total constant memory: " << deviceProp.totalConstMem / (1024 ) << " kB\n";

 
    //--- Освобождение памяти ---
    cudaFree(d_X);
    cudaFree(d_params);
    cudaFree(d_paramLinspaceA);


    return histEntropy2DFinal;
}