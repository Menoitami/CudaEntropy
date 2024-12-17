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

    if (inputString.size() < 4 || 
        (inputString.substr(inputString.size() - 4) != ".csv" &&
         inputString.substr(inputString.size() - 4) != ".txt")) {
        std::cerr << "Error: File must have a .csv or .txt extension!" << std::endl;
        return 1;
    }

    std::ofstream outFile(inputString);

    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return 1;
    }

    outFile.close();

   
    // Задаем параметры для расчета
    double transTime = 1000;  // Время переходного процесса
    double tMax = 200;       // Время моделирования после TT
    double h = 0.01;          // Шаг интегрирования

    std::vector<double> X = {37.0, -1.0, 1.0,-1.0}; // Начальное состояние
    int coord = 0;                        // Координата для анализа

    std::vector<double> params = {0, 0.1, 4}; // Параметры модели
    

    double startBin = -100; // Начало гистограммы
    double endBin = 100;    // Конец гистограммы
    double stepBin = 0.01;  // Шаг бинов гистограммы

    // Параметры для linspace
    double linspaceStartA = 3.5;  // Начало диапазона параметра
    double linspaceEndA = 5;   // Конец диапазона параметра
    int linspaceNumA = 2000;       // Количество точек параметра
    int paramNumberA = 1;         // Индекс параметра для анализа


    double linspaceStartB = 0;  // Начало диапазона параметра
    double linspaceEndB = 0.35;   // Конец диапазона параметра
    int linspaceNumB = 2000;
    int paramNumberB = 2;         // Индекс параметра для анализа

    auto start = std::chrono::high_resolution_clock::now();

    //Вызов функции histEntropyCUDA3D
    std::vector<std::vector<double>> histEntropy3D = histEntropyCUDA3D(
                                        transTime, tMax, h,
                                        X, coord,
                                        params, paramNumberA,paramNumberB,
                                        startBin, endBin, stepBin,
                                        linspaceStartA,linspaceEndA, linspaceNumA, linspaceStartB,linspaceEndB, linspaceNumB
                                    );

    writeToCSV(histEntropy3D,linspaceNumA,linspaceNumB,inputString);

    //  std::vector<double> histEntropy2D = histEntropyCUDA2D(
    //                                     transTime, tMax, h,
    //                                     X, coord,
    //                                     params, paramNumberA,
    //                                     startBin, endBin, stepBin,
    //                                     linspaceStartA,linspaceEndA, linspaceNumA
    //                                 );

    // writeToCSV(histEntropy2D,linspaceNumA,inputString);



    std::cout<<"End of gpu part\n";


     auto stop = std::chrono::high_resolution_clock::now();

     std::chrono::duration<double> duration = stop - start;
     std::cout << "Program execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}