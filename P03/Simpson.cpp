#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include <iomanip>

using namespace std;

double f(const double x) {
    return acos(cos(x) / (1 + 2 * cos(x)));
}

double Simpson(unsigned int Nthrds, double n){
    const double b = M_PI / 2, a = 0;   // Integral from a = 0 to b = π/2                                                                                                                                                                                                                                                                                                                                                                                                                   
    const double h = (b - a) / n;
    double sum = 0.0;                   // Sum

    double* s = (double*)calloc(Nthrds, sizeof(double));

    if (s == NULL) {
        cerr << "Unable to allocate " << (Nthrds * sizeof(double)) << " bytes" << endl;
        return -1;
    }

    cout << "Nthrds: " << Nthrds << endl;

    auto start_time = chrono::steady_clock::now();

    #pragma omp parallel                // OpenMP Compiler Directive
    {                                   // OpenMP parallel block construct
        unsigned int istart, iend;
        int id;
        int N = int(n) / 2;             // Number of Intervals
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        id = omp_get_thread_num();
        istart = id * N / Nthrds;
        iend = (id + 1) * N / Nthrds;

        if (id == Nthrds - 1){
            iend = N;
        }

        #pragma omp critical
        {
            cout << "id: " << id << ", istart: " << istart << ", iend: " << iend << endl;
        }

        for (unsigned long long j = istart + 1; j <= iend; ++j)
        {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            s[id] += f(a + (2 * j - 2) * h) + 4 * f(a + (2 * j - 1) * h) + f(a + (2 * j) * h);
        }
        s[id] *= h / 3;
    }

    auto start_reduction = chrono::steady_clock::now();

    #pragma omp parallel for reduction(+ : sum)
    for (int j = 0; j < Nthrds; ++j) {
        sum += s[j];
    }

    auto end_reduction = chrono::steady_clock::now();

    auto run_time = chrono::steady_clock::now();

    double exact_value = 5 * pow(M_PI, 2) / 24;
    double error = fabs(sum - exact_value);

    chrono::duration<double> elapsed_time = run_time - start_time;  // Compute elapsed time in seconds with decimals                                                                                                                                                                                                                                                                                                                                                                        

    cout << "approximation: " << scientific << setprecision(6) << sum
         << ", exact: " << scientific << setprecision(6) << exact_value
         << ", error: " << scientific << setprecision(6) << error
         << ", intervals: " << scientific << setprecision(0) << n
         << ", runtime: " << defaultfloat << setprecision(6) << elapsed_time.count() << " s"
         << ", threads: " << setw(3) << setfill('0') << Nthrds
         << endl;

    free(s);

    return sum;
}

int main(int argc, char *argv[]){
    unsigned int Nthrds;
    double n;

    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <num_Nthrds> <num_n>" << endl;
        return 1;
    }

    sscanf(argv[1], "%u", &Nthrds);
    sscanf(argv[2], "%lf", &n);

    cout << endl;
    cout << "requesting " << Nthrds << " threads" << endl;
    cout << "requesting " << scientific << setprecision(1) << n << " intervals" << endl;

    omp_set_dynamic(0);
    omp_set_num_threads(Nthrds);

    Simpson(Nthrds, n);

    cout << endl;

    return 0;
}