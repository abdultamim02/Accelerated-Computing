#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define T(i, j) (T[(i) * (n_cells + 2) + (j)])
#define T_new(i, j) (T_new[(i) * (n_cells + 2) + (j)])

double MAX_RESIDUAL = 1.e-8;

void kernel_omp_cpu(double *T, int max_iterations, int n_cells, double *residual, double *parallel_time) {
    double start_time;
    int iteration = 0;
    *residual = 1.e5;
    
    double *T_new = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));

    if (!T_new){
        printf("Memory Allocation Failed For T_new (Parallel)\n");
        exit(1);
    }

    start_time = omp_get_wtime();

    // simulation iterations
    while (*residual > MAX_RESIDUAL && iteration <= max_iterations) {
        if (iteration % 100 == 0) {  // Print every 100 iterations
            printf("Iteration: %d, Residual: %.9e\n", iteration, *residual);
            fflush(stdout);  // Force output to show immediately
        }

        // main computational kernel, average over neighbours in the grid 
        #pragma omp parallel for collapse(2) 
            for (unsigned i = 1; i <= n_cells; i++)
            {
                for (unsigned j = 1; j <= n_cells; j++)
                {
                    T_new(i, j) = 0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
                }
            }

            // Reset residual
            double local_residual = 0.0;

        // compute the largest change and copy T_new to T 
        #pragma omp parallel for reduction(max : local_residual) collapse(2) 
            for (unsigned int i = 1; i <= n_cells; i++) 
            { 
                for (unsigned int j = 1; j <= n_cells; j++) 
                { 
                    local_residual = MAX(fabs(T_new(i, j) - T(i, j)), local_residual); 
                    T(i, j) = T_new(i, j);
                }
            }
            *residual = local_residual;
            iteration++;
    }

    *parallel_time = omp_get_wtime() - start_time;

    // Free Allocated Memory
    free(T_new);
}

int main(int argc, char *argv[]){
    unsigned int n_cells, iter_count;
    double serial_residual, serial_time;
    double parallel_residual, parallel_time;
    double max_error;
    const double tolerance = 1e-6;
    int passed;
    int numThreads = omp_get_max_threads();

    if (argc < 3) {
        printf("Usage: %s <n_cells> <iter_count>\n", argv[0]);
        return 1;
    }

    sscanf(argv[1], "%u", &n_cells);
    sscanf(argv[2], "%u", &iter_count);

    switch (iter_count)
    {
    case 100:
        serial_residual = 2.421390771e-03;
        serial_time = 0.942906;
        break;
    case 1000:
        serial_residual = 2.419262863e-04;
        serial_time = 9.507495;
        break;
    case 10000:
        serial_residual = 2.419687285e-05;
        serial_time = 94.682115;
        break;
    case 100000:
        serial_residual = 2.330164395e-06;
        serial_time = 940.961615;
        break;
    case 1000000:
        serial_residual = 2.899418128e-08;
        serial_time = 9364.358466;
        break;
    
    default: 
        fprintf(stderr, "Error: Invalid iter_count input: %d\n", iter_count);
        exit(1);
    }

    double *T = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));

    if (!T){
        printf("Memory Allocation Failed For T\n");
        exit(1);
    }

    // initialize grid and boundary conditions
    for (unsigned i = 0; i <= n_cells + 1; i++){
        for (unsigned j = 0; j <= n_cells + 1; j++){
            if((j == 0) || (j == (n_cells + 1))){
                T(i, j) = 1.0;
            }
            else {
                T(i, j) = 0.0;
            }
        }
    }

    // Run Parallel Implementation
    kernel_omp_cpu(T, iter_count, n_cells, &parallel_residual, &parallel_time);

    max_error = fabs(serial_residual - parallel_residual);
    passed = max_error < tolerance;

    printf("Using number of cells = %d\n", n_cells);
    printf("Using maximum iterations count = %d\n", iter_count);
    printf("Serial Residual = %.9e\n", serial_residual);
    printf("Residual = %.9e\n", parallel_residual);
    printf("Validation maximum error = %.9e : %s\n", max_error, passed ? "PASSED" : "FAILED");
    printf("CPU Serial kernel time = %lf Sec\n", serial_time);
    printf("OpenMP CPU time = %lf Sec\n", parallel_time);
    printf("Number of OpenMP threads = %d\n", numThreads);
    printf("Speedup = %lf\n", serial_time / parallel_time);

    // Free Allocated Memory                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    free(T);
    
    return 0;
}