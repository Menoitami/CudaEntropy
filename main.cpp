#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <list>
#include <stdexcept>
#include <functional>
#include <cstring>

const double EPSD = std::numeric_limits<double>::epsilon();

std::vector<double> linspace(double start, double end, int num) {
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

double calculateEntropy(const std::vector<double>& P) {

    double entropy = 0.0;

    for (const auto& p : P) {
        if (p > 0) {  
            entropy -= (p - EPSD) * std::log2(p);  
        }
    }

    return entropy;
}

void calculateDiscreteModel(double* x, const double* a, const double h)
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


double* RunSystem(
    void (*systemStep)(double*, const double*, const double),
    const double& totalTime, const double& stepSize, double* X, const double* a) {
    
    if (totalTime <= 0) throw std::invalid_argument("totalTime <= 0");
    if (stepSize <= 0) throw std::invalid_argument("stepSize <= 0");

    int iterations = static_cast<int>(totalTime / stepSize);

    for (int i = 0; i < iterations; ++i) {
        systemStep(X, a, stepSize);
    }

    return X;
}

std::vector<double> system_CD_findPeaks_Entropy(
    void (*systemStep)(double*, const double*, const double),
    const double& totalTime, const double& stepSize, double* X, const double* a, int coord,
    double binStart, double binEnd, double binStep, int& sum) {

    if (binStart >= binEnd) throw std::invalid_argument("binStart >= binEnd");
    if (binStep <= 0) throw std::invalid_argument("binStep <= 0");
    if (totalTime <= 0) throw std::invalid_argument("totalTime <= 0");
    if (stepSize <= 0) throw std::invalid_argument("stepSize <= 0");

    int iterations = static_cast<int>(totalTime / stepSize);
    double last = X[coord];
    bool lastBigger = false;

    int num_bins = static_cast<int>(std::ceil((binEnd - binStart) / binStep));
    std::vector<double> bins(num_bins, 0);

    for (int i = 0; i < iterations; ++i) {
        systemStep(X, a, stepSize);

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

    return bins;
}

int main() {
   void (*systemStep)(double*, const double*, const double) = calculateDiscreteModel;

    int coord = 0;
    double transTime = 1000;
    double tMax = 2000;
    double h = 0.01;

    double start[] = {0.1, 0.1, 0}; 
    int startSize = sizeof(start) / sizeof(start[0]); 
    double* X = new double[startSize];
    std::memcpy(X, start, startSize * sizeof(double));
    
    double params[] = {0, 0.2, 0.2, 5.7};

    double startEdge = -20;
    double endEdge = 20;
    double stepEdge = 0.1;

    std::vector<double> a_array = linspace(0.1, 0.35, 400);

    std::vector<double> HistEntropy;
    HistEntropy.reserve(a_array.size());
    std::vector<double> bins;

    for (double i : a_array) {
        std::memcpy(X, start, startSize * sizeof(double));
        params[1] = i;

        RunSystem(systemStep, transTime, h, X, params);

        int sum = 0;
        bins =std::move( system_CD_findPeaks_Entropy(
            systemStep, tMax, h, X, params, coord, startEdge, endEdge, stepEdge, sum));

        for (auto& pair : bins) {
            pair = pair / (sum + EPSD);
        }

        double H = calculateEntropy(bins);
        HistEntropy.push_back(H / std::log2(bins.size()));
    }

    for (double num : HistEntropy) {
        std::cout << num << " ";
    }

    delete[] X;
    return 0;
}