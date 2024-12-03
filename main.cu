#include <iostream>
#include <vector>
#include "lib.cuh"
#include <string>


int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "  file.csv to write\n";
        return 1;
    }

    std::string inputString = argv[1];
   
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
    double linspaceEndB = 0.2;   // Конец диапазона параметра
    int linspaceNumB = 100;
    std::vector<double> paramLinspaceB = linspace(linspaceStartB, linspaceEndB, linspaceNumB);
    int paramNumberB = 2;         // Индекс параметра для анализа

    //Вызов функции histEntropyCUDA3D
    std::vector<std::vector<double>> histEntropy3D = histEntropyCUDA3D(
                                        transTime, tMax, h,
                                        X, coord,
                                        params, paramNumberA,paramNumberB,
                                        startBin, endBin, stepBin,
                                        paramLinspaceA, paramLinspaceB
                                    );

    writeToCSV(histEntropy3D,linspaceNumB,linspaceNumA,inputString);

    //Вызов функции histEntropyCUDA2D
    // std::vector<double> histEntropy2D = histEntropyCUDA2D(
    //                                     transTime, tMax, h,
    //                                     X, coord,
    //                                     params, paramNumberA,
    //                                     startBin, endBin, stepBin,
    //                                     paramLinspaceA
    //                                 );

    // writeToCSV(histEntropy2D,1,linspaceNumA,inputString);
    
    return 0;
}