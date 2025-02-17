#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <omp.h>
#include <chrono>

using namespace std;

double f(const double x) {
    return 1.0 / (1.0 + x * x);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <num_threads>" << endl;
        return -1;
    }

    unsigned int Nthrds;
    sscanf(argv[1], "%u", &Nthrds);
    cout << "Requesting " << Nthrds << " threads" << endl;

    omp_set_dynamic(0);
    omp_set_num_threads(Nthrds);

    const unsigned int n = (unsigned int)1e9;
    const double b = 1, a = 0;
    const double h = (b - a) / n;
    double S = 0;
    double exact = M_PI / 4;

    double *A = (double *)calloc(n + 1, sizeof(double));
    if (A == NULL) {
        cerr << "Unable to allocate " << (n + 1) * sizeof(double) << " bytes" << endl;
        return -1;
    }

    auto start_time = chrono::steady_clock::now();

    #pragma omp parallel
    {
        unsigned long long istart, iend;
        int id = omp_get_thread_num();
        istart = id * n / Nthrds;
        iend = (id + 1) * n / Nthrds;
        if (id == Nthrds - 1) iend = n;

        for (unsigned long long i = istart; i < iend; ++i) {
            A[i] = f(a + i * h);
        }
    }

    auto start_reduction = chrono::steady_clock::now();

    #pragma omp parallel for reduction(+ : S)
    for (int j = 0; j <= n; ++j) {
        S += A[j];
    }

    auto end_reduction = chrono::steady_clock::now();
    S *= h;

    auto end_time = chrono::steady_clock::now();

    cout << "Approximation: " << scientific << S
         << ", error: " << fabs(S - exact) << endl;

    cout << "Exact (pi/4): " << M_PI / 4 << endl;
    cout << "Elapsed time: "
         << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count()
         << " [us]" << endl;

    cout << "Reduction elapsed time: "
         << chrono::duration_cast<chrono::microseconds>(end_reduction - start_reduction).count()
         << " [us]" << endl;

    free(A);
    return 0;
}