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
#include <iomanip>
#include <algorithm>

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

__device__ double atomicExchDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    return __longlong_as_double(old);
}
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

    for (double value = start; value <= end; value += step) {
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


    for (int i = 0; i < iterations; ++i) {
        calculateDiscreteModel(X, param, d_h);

        if (X[d_coord] > last) {
            lastBigger = true;
        } else if (X[d_coord] < last && lastBigger) {
            if (last >= d_startBin && last < d_endBin) {
                
                int index = static_cast<int>((last - d_startBin) / d_stepBin);
                if (index < binSize && index >=0){
                bins[index]++;
                }
                sum++;
                
            }
            lastBigger = false;
        }
        last = X[d_coord];
    }

}


__global__ void calculateHistEntropyCuda3D(const double* X, 
                                           const double* params,
                                           const double* paramLinspaceA,
                                           const double* paramLinspaceB,
                                           double* histEntropy, 
                                           double* bins_global) {
    extern __shared__ double shared_mem[];

    // Shared memory allocation
    double* X_sh = shared_mem;
    double* params_sh = &shared_mem[d_XSize];

    // Load global memory to shared memory
    for (int i = threadIdx.x; i < d_XSize; i += blockDim.x) {
        X_sh[i] = X[i];
    }
    for (int i = threadIdx.x; i < d_paramsSize; i += blockDim.x) {
        params_sh[i] = params[i];
    }
    
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_histEntropySize-1) return;


    int binSize = static_cast<int>(std::ceil((d_endBin - d_startBin) / d_stepBin));
    int offset = idx * binSize;
    if (offset + binSize > d_histEntropySize * binSize) return;
    double* bin = &bins_global[offset];



    // Compute row and column indices
    int row = idx / d_histEntropySizeCol;
    int col = idx % d_histEntropySizeCol;

    // Local variables for computation
    double X_locals[32];
    double params_local[32];

    memcpy(X_locals, X_sh, d_XSize * sizeof(double));
    memcpy(params_local, params_sh, d_paramsSize * sizeof(double));

    params_local[d_paramNumberA] = paramLinspaceA[row];
    params_local[d_paramNumberB] = paramLinspaceB[col];

    // Perform calculations
    loopCalculateDiscreteModel(X_locals, params_local, d_h, 
                               static_cast<int>(d_transTime / d_h), 
                               0, 0, 0, nullptr, 0, 0);

    int sum = 0;

    CalculateHistogram(X_locals, params_local, sum, bin);
    double H = calculateEntropy(bin, binSize, sum);

    // Normalize and store result
    histEntropy[row * d_histEntropySizeCol + col] = (H / __log2f(binSize));

    // Atomic progress update (optional)
    if ((atomicAdd(&d_progress, 1) + 1) % (d_histEntropySize / 10) == 0) {
        printf("Progress: %d%%\n", ((d_progress + 1) * 100) / d_histEntropySize);
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
    //    if (linspaceStartB == linspaceEndB) throw std::invalid_argument("linspaceStartB == linspaceEndB");
    //    if (linspaceStartA == linspaceEndA) throw std::invalid_argument("linspaceStartA == linspaceEndA");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return std::vector<std::vector<double>>(0); 
    }

    int device = 0;
    cudaDeviceProp deviceProp;
    int numBlocks;
    int threadsPerBlock;

    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, device));
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock/4;

    int XSize = X.size();
    int paramsSize = params.size();
    int histEntropySizeRow = linspaceNumA;
    int histEntropySizeCol =  linspaceNumB;
    int histEntropySize = histEntropySizeRow * histEntropySizeCol;
    int binSize = static_cast<int>(std::ceil((endBin - startBin) / stepBin));
    std::vector<double> paramLinspaceA = linspaceNum(linspaceStartA, linspaceEndA,linspaceNumA);
    std::vector<double> paramLinspaceB = linspaceNum(linspaceStartB, linspaceEndB,linspaceNumB);
    
     // --- Выделение памяти на устройстве ---
    double *d_X, *d_params, *d_paramLinspaceA;

    CHECK_CUDA_ERROR(cudaMalloc(&d_X, XSize * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_params, paramsSize * sizeof(double)));
    

    CHECK_CUDA_ERROR(cudaMemcpy(d_X, X.data(), XSize * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_params, params.data(), paramsSize * sizeof(double), cudaMemcpyHostToDevice));
    


    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_startBin, &startBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_endBin, &endBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_stepBin, &stepBin, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tMax, &tMax, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_transTime, &transTime, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_h, &h, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_coord, &coord, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_XSize, &XSize, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramsSize, &paramsSize, sizeof(int)));

    std::cout<<XSize<<" "<<paramsSize<<"\n";
    
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramNumberA, &paramNumberA, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_paramNumberB, &paramNumberB, sizeof(int)));

    size_t freeMem, totalMem;

	CHECK_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
    freeMem*=0.7/1024/1024; //bytes


    double bytes = static_cast<double>(histEntropySize)*sizeof(double)/(1024 * 1024)+static_cast<double>(histEntropySize)*static_cast<double>(binSize)*sizeof(double)/(1024 * 1024);
    
    int iteratationsB = 1;
    int iteratationsA = 1;

    double memEnabledB = linspaceNumB;
    double memEnabledA = linspaceNumA;

    while ((((memEnabledA*std::ceil(memEnabledB)+ static_cast<double>(binSize)*memEnabledA*std::ceil(memEnabledB))* sizeof(double)))/(1024*1024) > freeMem ){
        if (std::ceil(memEnabledB) <= 1){
            memEnabledA /=2;
            iteratationsA*=2;
        }
        else{
            memEnabledB /=2;
            iteratationsB*=2;
        }
    }

    memEnabledA =std::ceil(memEnabledA);
    memEnabledB =std::ceil(memEnabledB);

    std::cout<<"freeMem: "<<freeMem<<"MB Needed Mbytes: "<<bytes<<"MB\n";
    std::cout<<"iterationsB: "<<iteratationsB<<" B_size: "<<" "<<memEnabledB<<"\n";
    std::cout<<"iterationsA: "<<iteratationsA<<" A_size: "<<" "<<memEnabledA<<"\n";

    std::vector<std::vector<double>> histEntropy2DFinal;
    histEntropy2DFinal.reserve(histEntropySizeCol);
    int SizeCol;
    double remainingB;
    double currentStepB;
    double processedB = 0;
    
    for (int i = 0 ; i<iteratationsB;++i){
        remainingB = histEntropySizeCol - processedB;
        currentStepB = std::min(memEnabledB, remainingB);


        std::vector<double> partParamLinspaceB(currentStepB);

        std::copy(
            paramLinspaceB.begin() + processedB,
            paramLinspaceB.begin() + processedB + currentStepB,
            partParamLinspaceB.begin()
        );

        processedB += currentStepB;
        SizeCol = partParamLinspaceB.size();

        double *d_paramLinspaceB, *d_histEntropy;




        int SizeRow;
        double remainingA;
        double currentStepA;
        double processedA = 0;

        std::vector<double> histEntropyRowFinal;

        for (int j = 0 ; j<iteratationsA;++j){
            remainingA = histEntropySizeRow - processedA;
            currentStepA = std::min(memEnabledA, remainingA);


            std::vector<double> partParamLinspaceA(currentStepA);

            std::copy(
                paramLinspaceA.begin() + processedA,
                paramLinspaceA.begin() + processedA + currentStepA,
                partParamLinspaceA.begin()
            );

            processedA += currentStepA;
            SizeRow = partParamLinspaceA.size();

            histEntropySize = SizeRow * SizeCol;
            std::vector<double> histEntropy(histEntropySize);

            CHECK_CUDA_ERROR(cudaMalloc(&d_histEntropy, histEntropySize * sizeof(double)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_paramLinspaceB, SizeCol * sizeof(double)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_paramLinspaceA, SizeRow * sizeof(double)));

            CHECK_CUDA_ERROR(cudaMemcpy(d_paramLinspaceB, partParamLinspaceB.data(), SizeCol * sizeof(double),cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_paramLinspaceA, partParamLinspaceA.data(), SizeRow * sizeof(double), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySize, &histEntropySize, sizeof(int)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySizeCol, &SizeCol, sizeof(int)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_histEntropySizeRow, &SizeRow, sizeof(int)));

            double* bins;
            cudaMalloc(&bins, (histEntropySize * binSize) * sizeof(double));
            cudaMemset(bins, 0, histEntropySize * binSize * sizeof(double));

            numBlocks = deviceProp.multiProcessorCount * 4;
            threadsPerBlock = std::ceil(histEntropySize / (float)numBlocks);

            if (threadsPerBlock > maxThreadsPerBlock) {
                threadsPerBlock = maxThreadsPerBlock;
                numBlocks = std::ceil(histEntropySize / (float)threadsPerBlock);
            }
            else if (threadsPerBlock ==0) threadsPerBlock =1;   


            std::cout << "Memory block is: " << i << " / " << iteratationsB  << "\n";
            std::cout<<"blocks: "<<numBlocks<<" threads: "<<threadsPerBlock<<" sm's: " << deviceProp.multiProcessorCount<<"\n";


            int progress = 0;
            cudaMemcpyToSymbol(d_progress, &progress, sizeof(int));

            calculateHistEntropyCuda3D<<<numBlocks, threadsPerBlock, (128)*sizeof(double)*2>>>(
            d_X, d_params, d_paramLinspaceA, d_paramLinspaceB, d_histEntropy,bins);

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
                throw std::runtime_error("CUDA kernel execution failed");
            }
            std::vector<std::vector<double>> histEntropy2D(SizeCol, std::vector<double>(SizeRow));
            cudaMemcpy(histEntropy.data(), d_histEntropy, histEntropySize * sizeof(double), cudaMemcpyDeviceToHost);

            

            if (memEnabledB == 1){
                histEntropyRowFinal.insert(histEntropyRowFinal.end(), histEntropy.begin(), histEntropy.end()); 
            }
            else{
                for (int q = 0; q < SizeRow; ++q) {
                    for (int s = 0; s < SizeCol; ++s) {
                        histEntropy2D[s][q] = std::move(histEntropy[q * SizeCol + s]);
                    }
                }
                for (int q = 0; q < SizeCol; ++q) {
                    histEntropy2DFinal.push_back(histEntropy2D[q]);
                }
            }

            cudaFree(bins);
            cudaFree(d_paramLinspaceA);
        }

        if (memEnabledB == 1){
            histEntropy2DFinal.push_back(histEntropyRowFinal);
        }
        cudaFree(d_paramLinspaceB);

    }


    // cudaMemGetInfo(&freeMem, &totalMem);
    // std::cout << "Free memory: " << freeMem / (1024 * 1024) << " MB\n";
    // std::cout << "Total memory: " << totalMem / (1024 * 1024) << " MB\n";
    // std::cout << "Total constant memory: " << deviceProp.totalConstMem / (1024 ) << " kB\n";

 
    //--- Освобождение памяти ---
    cudaFree(d_X);
    cudaFree(d_params);


    return histEntropy2DFinal;
}



__global__ void calculateHistEntropyCuda2D(const double* const X, 
                                           const double* const params,
                                           const double* const paramLinspaceA,
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


        double X_locals[x_size];
        double params_local[param_size];
        
        memcpy(params_local, params_sh, param_size * sizeof(double));
        memcpy(X_locals, X_sh, x_size * sizeof(double));

        params_local[d_paramNumberA] = paramLinspaceA[idx];

        loopCalculateDiscreteModel(X_locals, params_local, d_h, 
                                   static_cast<int>(d_transTime / d_h), 
                                   0, 0, 0, nullptr, 0, 0);

        int binSize = static_cast<int>(ceil((d_endBin - d_startBin) / d_stepBin));
        int sum = 0;

        CalculateHistogram(
             X_locals, params_local, sum, bins_global[idx]);

        double H = calculateEntropy(bins_global[idx], binSize, sum);

        // Нормализуем и сохраняем результат в глобальную память
        histEntropy[idx] = (H / __log2f(binSize));


        // Вывод прогресса
        
        int progress = atomicAdd(&d_progress, 1);
        if ((progress + 1) % (d_histEntropySize / 10) == 0) {
            printf("Progress: %d%%\n", ((progress + 1) * 100) / d_histEntropySize);
        }
        

    }
}


__host__ std::vector<double> histEntropyCUDA2D(
    const double transTime,const double tMax,const double h, 
    const std::vector<double>& X,const int coord, 
    std::vector<double>& params,const int paramNumberA,
    const double startBin, const double endBin,const double stepBin, 
    double linspaceStartA, double linspaceEndA, int linspaceNumA
)
 {
    params.push_back(0);
    int paramNumberB = params.size()-1;
    double linspaceStartB = params[paramNumberB];
    double linspaceEndB = linspaceStartB;
    double linspaceNumB = 1;
    std::vector<std::vector<double>> histEntropy3D = histEntropyCUDA3D(transTime,tMax,h,X,coord,params,paramNumberA,paramNumberB,
                                                                        startBin,endBin,stepBin,linspaceStartA,linspaceEndA,linspaceNumA,
                                                                        linspaceStartB,linspaceEndB,linspaceNumB);

    return histEntropy3D[0];
}