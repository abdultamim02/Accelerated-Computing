#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define T(i, j) (T[(i) * (n_cells + 2) + (j)])
#define T_new(i, j) (T_new[(i) * (n_cells + 2) + (j)])

double MAX_RESIDUAL = 1.e-8;

void kernel_omp_gpu_teams_simd(double *T, int max_iterations, int n_cells, double *residual, double *gpu_simd_time) {
    double start_time;
    int iteration = 0;
    *residual = 1.e5;

    double *T_new = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));

    if (!T_new) {
        printf("Memory Allocation Failed For T_new (GPU)\n");
        exit(1);
    }

    start_time = omp_get_wtime();

    // Ensure correct data transfer to the GPU
    #pragma omp target enter data map(to: T[0:(n_cells+2)*(n_cells+2)]) map(alloc: T_new[0:(n_cells+2)*(n_cells+2)])

    while (*residual > MAX_RESIDUAL && iteration <= max_iterations) {
        if (iteration % 100 == 0) {
            printf("Iteration: %d, Residual: %.9e\n", iteration, *residual);
            fflush(stdout);
        }

        // GPU Teams with SIMD
        #pragma omp target teams distribute parallel for simd collapse(2)
        for (unsigned i = 1; i <= n_cells; i++) {
            for (unsigned j = 1; j <= n_cells; j++) {
                T_new(i, j) = 0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
            }
        }

        double local_residual = 0.0;

        // Compute residual with SIMD parallelization
        #pragma omp target teams distribute parallel for simd reduction(max: local_residual) collapse(2)
        for (unsigned int i = 1; i <= n_cells; i++) {
            for (unsigned int j = 1; j <= n_cells; j++) {
                local_residual = MAX(fabs(T_new(i, j) - T(i, j)), local_residual);
                T(i, j) = T_new(i, j);
            }
        }

        *residual = local_residual;
        iteration++;
    }

    #pragma omp target exit data map(from: T[0:(n_cells+2)*(n_cells+2)])

    *gpu_simd_time = omp_get_wtime() - start_time;

    free(T_new);
}

int main(int argc, char *argv[]){
    unsigned int n_cells, iter_count;
    double gpu_simd_residual, gpu_simd_time;
    double serial_time;
    double parallel_time;
    double gpu_time;

    if (argc < 3) {
        printf("Usage: %s <n_cells> <iter_count>\n", argv[0]);
        return 1;
    }

    sscanf(argv[1], "%u", &n_cells);
    sscanf(argv[2], "%u", &iter_count);

    switch (iter_count)
    {
    case 100:
        serial_time = 0.942906;
        break;
    case 1000:
        serial_time = 9.507495;
        break;
    case 10000:
        serial_time = 94.682115;
        parallel_time = 48.397173;
        gpu_time = 293.151529;
        break;
    case 100000:
        serial_time = 940.961615;
        break;
    case 1000000:
        serial_time = 9364.358466;
        parallel_time = 8439.665795;
        gpu_time = 5224.007010;
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

    // Run gpu teams Implementation
    kernel_omp_gpu_teams_simd(T, iter_count, n_cells, &gpu_simd_residual, &gpu_simd_time);

    printf("Using number of cells = %d\n", n_cells);
    printf("Using maximum iterations count = %d\n", iter_count);
    printf("CPU Serial kernel time = %lf Sec\n", serial_time);
    printf("OpenMP CPU time = %lf Sec\n", parallel_time);
    printf("OpenMP GPU teams (without vectorization) time = %lf Sec\n", gpu_time);
    printf("OpenMP GPU teams (with SIMD) time = %lf Sec\n", gpu_simd_time);
    printf("GPU Simd Residual = %.9e\n", gpu_simd_residual);
    printf("Speedup (GPU SIMD vs. CPU) = %lf\n", gpu_time / gpu_simd_time);

    // Free Allocated Memory                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    free(T);
    
    return 0;
}