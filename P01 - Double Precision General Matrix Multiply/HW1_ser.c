#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ORDER 1000
#define AVAL 3.0
#define BVAL 5.0

int main(int argc, char *argv[])
{
    int Ndim, Pdim, Mdim; 
    int i, j, k;
    double *A, *B, *C;

    double start_time, run_time, dN, tmp, mflops, errsq = 0.0, expected_value;

    Ndim = ORDER;
    Pdim = ORDER;
    Mdim = ORDER;

    /* A[N][P], B[P][M], C[N][M] */
    A = (double *)malloc(Ndim * Pdim * sizeof(double));
    B = (double *)malloc(Pdim * Mdim * sizeof(double));
    C = (double *)malloc(Ndim * Mdim * sizeof(double));

    /* Initialize matrices */
    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Pdim; j++)
            *(A + (i * Ndim + j)) = AVAL;

    for (i = 0; i < Pdim; i++)
        for (j = 0; j < Mdim; j++)
            *(B + (i * Pdim + j)) = BVAL;

    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Mdim; j++)
            *(C + (i * Ndim + j)) = 0.0;

    start_time = omp_get_wtime();

    for (i = 0; i < Ndim; i++){
        for (j = 0; j < Mdim; j++){
            tmp = 0.0;
            for(k = 0;k < Pdim; k++){
                /* C(i, j) = sum(over k) A(i, k) * B(k, j) */
                tmp += *(A + (i * Ndim + k)) * *(B + (k * Pdim + j));
            }
        *(C + (i * Ndim + j)) = tmp;
        }
    }

    run_time = omp_get_wtime() - start_time;
    dN = (double)ORDER;
    mflops = 2.0 * dN * dN * dN/(1000000.0 * run_time);

    printf("Order %d multiplication in %f seconds\n %f mflops\n\n", ORDER, run_time, mflops);

    /* Compute Error */
    expected_value = AVAL * BVAL * Pdim;
    for (i = 0; i < Ndim; i++) {
        for (j = 0; j < Mdim; j++) {
            double diff = *(C + (i * Ndim + j)) - expected_value;
            errsq += diff * diff;
        }
    }

    printf("No errors (errsq = %.16e).\n", errsq);

    free(A);
    free(B);
    free(C);

    return 0;
}