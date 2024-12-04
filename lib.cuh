#ifndef LIB_CUH
#define LIB_CUH

#ifdef _WIN32
  #ifdef CUDALIB_EXPORTS
    #define CUDA_API __declspec(dllexport)  // Экспорт символов для .dll
  #else
    #define CUDA_API __declspec(dllimport)  // Импорт символов из .dll
  #endif
#else
  #define CUDA_API  // Для других систем (не Windows), экспортировать без изменений
#endif

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <fstream>
#include <string>


__device__ const double EPSD = std::numeric_limits<double>::epsilon();


CUDA_API __host__ std::vector<double> linspace(double start, double end, int num);

CUDA_API __host__ void writeToCSV(const std::vector<std::vector<double>>& histEntropy2D, int cols, int rows, const std::string& filename);

CUDA_API std::vector<std::vector<double>> convert1DTo2D(const std::vector<double>& histEntropy1D);

CUDA_API __host__ void writeToCSV(const std::vector<double>& histEntropy1D, int cols, const std::string& filename);

CUDA_API __device__ __host__ void calculateDiscreteModel(double* x, const double* a, const double h);

CUDA_API __device__ __host__ bool loopCalculateDiscreteModel(double* x, const double* values, const double h,
 const int amountOfIterations, const int preScaller,
 int writableVar, const double maxValue, double* data, const int startDataIndex, const int writeStep);

CUDA_API __device__ __host__ void CalculateHistogram(
    const double& totalTime, const double& stepSize, double* X, const double* param, int coord,
    double binStart, double binEnd, double binStep, int& sum, double* bins);


CUDA_API __global__ void calculateHistEntropyCuda2D(const double* const X,const int* XSize, const double* const params,
                                            const int* paramsSize, const int* paramNumber,const double* const paramLinspace,
                                            int* coord, double* tMax, double* transTime, double* h, double* startBin,
                                            double* endBin, double* stepBin, double* histEntropy, int* histEntropySize , double** bins_global);

CUDA_API __global__ void calculateHistEntropyCuda3D(const double* const X,const int* XSize, const double* const params,
                                            const int* paramsSize, const int* paramNumberA, const int* paramNumberB, const double* const paramLinspaceA,
                                            const double* const paramLinspaceB, const int* histEntropySizeRow,  const int* histEntropySizeCol,
                                            int* coord, double* tMax, double* transTime, double* h, double* startBin, double* endBin,
                                            double* stepBin, double* histEntropy, int* histEntropySize , double** bins_global);
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
     
CUDA_API __host__ std::vector<double> histEntropyCUDA2D(
    const double transTime,const double tMax,const double h, 
    const std::vector<double>& X,const int coord, 
    const std::vector<double>& params,const int paramNumber, 
    const double startBin,const double endBin, const double stepBin, 
    const std::vector<double>& paramLinspace
);

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
     

CUDA_API __host__ std::vector<std::vector<double>> histEntropyCUDA3D(
    const double transTime,const double tMax,const double h, 
    const std::vector<double>& X,const int coord, 
    const std::vector<double>& params,const int paramNumberA,const int paramNumberB, 
    const double startBin, const double endBin,const double stepBin, 
    const std::vector<double>& paramLinspaceA,const std::vector<double>& paramLinspaceB
);

#endif // LIB_CUH