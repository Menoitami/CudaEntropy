#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>


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
    const double& totalTime, const double& stepSize, double* X, const double* a, int coord,
    double binStart, double binEnd, double binStep, int& sum, double* bins) {


    int iterations = static_cast<int>(totalTime / stepSize);
    double last = X[coord];
    bool lastBigger = false;

    int binSize = static_cast<int>(ceil((binEnd - binStart) / binStep));

    for (int i = 0; i < binSize; ++i) {
        bins[i] = 0.0;
    }


    for (int i = 0; i < iterations; ++i) {
        calculateDiscreteModel(X, a, stepSize);

        if (X[coord] > last) {
            lastBigger = true;
        } else if (X[coord] < last && lastBigger) {
            if (last >= binStart && last < binEnd) {
                int index = static_cast<int>((last - binStart) / binStep);
                bins[index]++;
                sum++;
            }
            lastBigger = false;
        }
        last = X[coord];
    }

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


__global__ void calculateHistEntropyCuda(const double* const X,const int* XSize, const double* const params,
                                            const int* paramsSize, const int* paramNumber,const double* const paramLinspace,
                                            int* coord, double* tMax, double* transTime, double* h, double* startBin,
                                            double* endBin, double* stepBin, double* histEntropy, int* histEntropySize , double** bins_global) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < *histEntropySize) {

        //выделение локальных параметров для потока
        double* params_local = new double[*paramsSize];
        memcpy(params_local, params, *paramsSize * sizeof(double));   

        params_local[*paramNumber] = paramLinspace[idx];

        //выделение координат на поток параметров для потока 
        double* X_local = new double[*XSize]; 
        memcpy(X_local, X, *XSize * sizeof(double));

        loopCalculateDiscreteModel(X_local, params_local, *h, static_cast<int>(*transTime / *h), 0, 0, 0,nullptr,0, 0);

        //самое проблемное место, выделение глобальной памяти в функции(мб можно создать 2d массив сразу в host функции)
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
     
__host__ std::vector<double> histEntropy2DCUDA(
    double transTime, double tMax, double h, 
    const std::vector<double>& X, int coord, 
    const std::vector<double>& params, int paramNumber, 
    double startBin, double endBin, double stepBin, 
    double linspaceStart, double linspaceEnd, int linspaceNum
)
 {
    // Проверки переменных
    if (tMax <= 0) throw std::invalid_argument("tMax <= 0");
    if (transTime <= 0) throw std::invalid_argument("transTime <= 0");
    if (h <= 0) throw std::invalid_argument("h <= 0");
    if (startBin >= endBin) throw std::invalid_argument("binStart >= binEnd");
    if (stepBin <= 0) throw std::invalid_argument("binStep <= 0");
    if (coord < 0 || coord >= X.size()) throw std::invalid_argument("coord out of range X");
    if (paramNumber < 0 || paramNumber >= params.size()) throw std::invalid_argument("paramNumber out of range params");
    int XSize = X.size();
    int paramsSize = params.size();
    std::vector<double> paramLinspace = linspace(linspaceStart, linspaceEnd, linspaceNum);
    int histEntropySize = paramLinspace.size();

    std::vector<double> HistEntropy(histEntropySize);

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
    
    
    int binMem = static_cast<int>(ceil((endBin - startBin) / stepBin))*sizeof(double);
    int paramMem = histEntropySize * sizeof(double);

    // int threadsPerBlock = 1024; // начальное значение
    // while (threadsPerBlock * (binMem + paramMem) > 49152) {
    //     threadsPerBlock /= 2;
    // }
    // int numBlocks = (histEntropySize + threadsPerBlock - 1) / threadsPerBlock;

    // --- Запуск CUDA-ядра ---
    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "Max grid dimensions (blocks per dimension):" << std::endl;
    std::cout << "  X: " << deviceProp.maxGridSize[0] << std::endl;
    std::cout << "  Y: " << deviceProp.maxGridSize[1] << std::endl;
    std::cout << "  Z: " << deviceProp.maxGridSize[2] << std::endl;
    // Память на блок:
    int sharedMemPerBlock = deviceProp.sharedMemPerBlock;

    int threadsPerBlock = deviceProp.maxThreadsPerBlock;

// Проверяем, помещается ли память на один блок:
/*
    while (threadsPerBlock * (binMem + paramMem) > sharedMemPerBlock) {
        threadsPerBlock /= 2;
    }
    */

// Вычисляем количество блоков:
    int numBlocks = (histEntropySize + threadsPerBlock - 1) / threadsPerBlock;


    // Выделяем память для bins как двумерный массив для всех потоков
    int binSize = static_cast<int>(ceil((endBin - startBin) / stepBin));
    
    double** hostBins = (double**)malloc((numBlocks * threadsPerBlock ) * sizeof(double*));
    for (int i = 0; i < numBlocks * threadsPerBlock; ++i) {
        cudaMalloc(&hostBins[i], binSize * sizeof(double));
    }

    double** bins;
    cudaMalloc(&bins, (numBlocks * threadsPerBlock) * sizeof(double*));
    cudaMemcpy(bins, hostBins, (numBlocks * threadsPerBlock) * sizeof(double*), cudaMemcpyHostToDevice);
    free(hostBins);


    std::cout<<sharedMemPerBlock<<" "<<numBlocks<<" "<<threadsPerBlock<<"\n";

    calculateHistEntropyCuda<<<numBlocks, threadsPerBlock>>>(d_X, d_XSize, d_params, d_paramsSize,
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

    for (double val : HistEntropy) {
        std::cout << val << " ";
    }

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

    return HistEntropy;
}


int main() {
    printf("90");
    // Задаем параметры для расчета
    double transTime = 1000;  // Время переходного процесса
    double tMax = 2000;       // Время моделирования после TT
    double h = 0.01;          // Шаг интегрирования

    std::vector<double> X = {0.1, 0.1, 0}; // Начальное состояние
    int coord = 0;                        // Координата для анализа

    std::vector<double> params = {0, 0.2, 0.2, 5.7}; // Параметры модели
    int paramNumber = 1;                             // Индекс параметра для анализа

    double startBin = -20; // Начало гистограммы
    double endBin = 20;    // Конец гистограммы
    double stepBin = 0.01;  // Шаг бинов гистограммы

    // Параметры для linspace
    double linspaceStart = 0.1;  // Начало диапазона параметра
    double linspaceEnd = 0.35;   // Конец диапазона параметра
    int linspaceNum = 8000;       // Количество точек параметра
    //Вызов функции histEntropy2DCUDA
    std::vector<double> histEntropy = histEntropy2DCUDA(
                                        transTime, tMax, h,
                                        X, coord,
                                        params, paramNumber,
                                        startBin, endBin, stepBin,
                                        linspaceStart, linspaceEnd, linspaceNum
                                    );


    

    // histEntropy2DCUDA(
    //     1000, 2000, 0.01,
    //     {0.1, 0.1, 0}, 0,
    //     {0, 0.2, 0.2, 5.7}, 1,
    //     -20, 20, 0.1,
    //     0.1, 0.35, 400
    // );

    return 0;
}