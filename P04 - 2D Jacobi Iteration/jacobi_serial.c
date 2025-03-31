#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define T(i, j) (T[(i) * (n_cells + 2) + (j)])
#define T_new(i, j) (T_new[(i) * (n_cells + 2) + (j)])

double MAX_RESIDUAL = 1.e-15;

void kernel(double *T, int max_iterations, int n_cells) {
    double start_time, run_time;
    int iteration = 0;
    double residual = 1.e6;

    double *T_new = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));

    if (!T_new){
        printf("Memory Allocation Failed For T_new\n");
        exit(1);
    }

    start_time = omp_get_wtime();

    while (residual > MAX_RESIDUAL && iteration < max_iterations) {
        for (unsigned i = 1; i <= n_cells; i++)
        {
            for (unsigned j = 1; j <= n_cells; j++)
            {
                T_new(i, j) = 0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
            }
        }

        residual = 0.0;

        for (unsigned int i = 1; i <= n_cells; i++)
        {
            for (unsigned int j = 1; j <= n_cells; j++)
            {
                residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
                T(i, j) = T_new(i, j);
            }
        }
        iteration++;
    }

    run_time = omp_get_wtime() - start_time;

    printf("%% Using number of cells = %d\n", n_cells);
    printf("%% Using maximum iterations count = %d\n", max_iterations);
    printf("%% Serial Residual = %.9e\n", residual);
    printf("%% CPU Serial kernel time = %lf Seconds\n\n", run_time);

    printf("T = [...\n");
    for (unsigned i = 0; i <= (n_cells + 1); i++) {
        for (unsigned j = 0; j <= (n_cells + 1); j++)
            printf("%f ",T(i, j));
        printf(";...\n");
    }
    printf("];\n");

    // Free Allocated Memory                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    free(T_new);
}

int main(int argc, char *argv[]){

    unsigned int n_cells, iter_count;

    if (argc < 3) {
        printf("Usage: %s <n_cells> <iter_count>\n", argv[0]);
        return 1;
    }

    sscanf(argv[1], "%u", &n_cells);
    sscanf(argv[2], "%u", &iter_count);

    double *T = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));
    if (!T){
        printf("Memory Allocation Failed For T_new\n");
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

    kernel(T, iter_count, n_cells);

    // Free Allocated Memory                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    free(T);

    return 0;
}