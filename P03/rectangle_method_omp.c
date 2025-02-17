#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double rectangle_method(double a, double b, int n){
    double x1, x2, width, height, area;
    double dx = (b - a) / n;
    double sum = 0.0;
    double start_time, run_time;

    start_time = omp_get_wtime();

    for (size_t i = 0; i < n; i++){
        x1 = a + i * dx;
        x2 = x1 + dx;
        width = dx;
        height = 1.0 / ((x1 + x2) / 2);
        area = width * height;
        sum += area;
    }

    run_time = omp_get_wtime() - start_time;

    double exact_value = log(b) - log(a);
    double error = fabs(sum - exact_value);

    printf("\n");
    printf("rect: N = \t%d, area = %.13lf, err = %.16e.\n", n, sum, error);
    printf("Elapsed time = %.06e s (Serial)\n", run_time);

    return sum;
}

double rectangle_method_omp(double a, double b, int N) {
    double start_time, run_time;
    double sum = 0.0;
    int numThreads = 1;

    // Determine the number of threads                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    #pragma omp parallel
    {
        #pragma omp single
        numThreads = omp_get_num_threads();
    }

    printf("numThreads: %d\n", numThreads);

    double* s = (double*)calloc(numThreads, sizeof(double));

    start_time = omp_get_wtime();

    #pragma omp parallel
    {
        double x1, x2, width, height, area;
        double dx = (b - a) / N;
        int id = omp_get_thread_num();
        int istart = id * N / numThreads;
        int iend = (id + 1) * N / numThreads;

        if (id == numThreads - 1) iend = N;

        for (size_t i = istart; i < iend; i++){
            x1 = a + i * dx;
            x2 = x1 + dx;
            width = dx;
            height = 1.0 / ((x1 + x2) / 2);
            area = width * height;
            s[id] += area;
        }
    }

    // Sum the partial results                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    for (size_t i = 0; i < numThreads; ++i){
        sum += s[i];
    }
    free(s);

    run_time = omp_get_wtime() - start_time;

    double exact_value = log(b) - log(a);
    double error = fabs(sum - exact_value);

    printf("rect: N = \t%d, area = %.13lf, err = %.16e.\n", N, sum, error);
    printf("Elapsed time = %.06e s (parallel, %d threads)\n", run_time, numThreads);
    printf("\n");

    return sum;
}

int main(int argc, char *argv[]){
    double a, b;
    int N;
    unsigned int n;

    if (argc < 5) {
        printf("Usage: %s <num_a> <num_b> <num_N> <num_n>\n", argv[0]);
        return 1;
    }

    sscanf(argv[1], "%lf", &a);
    sscanf(argv[2], "%lf", &b);
    sscanf(argv[3], "%d", &N);
    sscanf(argv[4], "%d", &n);

    omp_set_dynamic(0);
    omp_set_num_threads(n);

    rectangle_method(a, b, N);
    rectangle_method_omp(a, b, N);

    return 0;
}
