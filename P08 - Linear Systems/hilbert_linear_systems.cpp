#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

double *HilbertMatrix(int n)
{
    double *H = (double*)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            H[i * n + j] = (double)1.0 / ((i + 1) + (j + 1) - 1.0);
        }
    }
    return H;
}

double* B(int n)
{
    double *b = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++){
        b[i] = 1.0;
    }
    return b;
}

double* PerturbedB(int n){
    double *b = (double*)malloc(n * sizeof(double));
    srand(time(0));
    
    for (int i = 0; i < n; i++){
        double epsilon = (double)rand() / RAND_MAX; // [0, 1)
        if (epsilon == 0.0f){
            epsilon = 0.0001f; // ensure ε > 0
        }
        b[i] = 1.0f + epsilon;
    }
    return b;
}

void PrintMatrix(double *H, int n) {
    printf("H[%d] =\n", n);
    for (int i = 0; i < n; i++) {
        printf("\t| ");
        for (int j = 0; j < n; j++) {
            printf("%.7f", H[i * n + j]);  // 7 decimal places
            if (j < n - 1) {
                printf(", ");
            }
        }
        printf(" |\n");
    }
}

void PrintB(double *b, int n) {
    printf("b[%d] = [\n", n);
    for (int i = 0; i < n; i++) {
        printf("\t%.7f", b[i]);  // 7 decimal places
        if (i < n - 1) {
            printf("\n");
        }
    }
    printf("\n\t]\n");
}

void HilbertLinearSystems(cusolverDnHandle_t cusolverH, double* A, double* B, int m, bool doFactorization)
{
    const int lda = m;
    const int ldb = m;

    double* X = (double*)malloc(m * sizeof(double)); /* X = A\B */
    double* LU = (double*)malloc(lda * m * sizeof(double)); /* L and U */
    int* Ipiv = (int*)malloc(m * sizeof(int));      /* host copy of pivoting sequence */
    int info = 0;     /* host copy of error info */

    double *d_A = NULL; /* device copy of A */
    double *d_B = NULL; /* device copy of B */
    int *d_Ipiv = NULL; /* pivoting sequence */
    int *d_info = NULL; /* error info */
    int lwork = 0;      /* size of workspace */
    double *d_work = NULL; /* device workspace for getrf */
    const int pivot_on = 0;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;

    /* step 2: copy A to device */
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(double) * m);
    cudaStat3 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m);
    cudaStat4 = cudaMalloc ((void**)&d_info, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    /* timer setup */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /* step 3: query working space of getrf + LU factorization if requested */
    if (doFactorization) {
        status = cusolverDnDgetrf_bufferSize(
                                            cusolverH,
                                            m,
                                            m,
                                            d_A,
                                            lda,
                                            &lwork
                                            );
        assert(CUSOLVER_STATUS_SUCCESS == status);

        cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double) * lwork);
        assert(cudaSuccess == cudaStat1);

        if (pivot_on){
            status = cusolverDnDgetrf(
                                    cusolverH,
                                    m,
                                    m,
                                    d_A,
                                    lda,
                                    d_work,
                                    d_Ipiv,
                                    d_info
                                );
        }
        else {
            status = cusolverDnDgetrf(
                                    cusolverH,
                                    m,
                                    m,
                                    d_A,
                                    lda,
                                    d_work,
                                    NULL,
                                    d_info
                                );
        }
        cudaStat1 = cudaDeviceSynchronize();
        assert(CUSOLVER_STATUS_SUCCESS == status);
        assert(cudaSuccess == cudaStat1);

        if (pivot_on){
            cudaStat1 = cudaMemcpy(Ipiv , d_Ipiv, sizeof(int) * m, cudaMemcpyDeviceToHost);
        }
        cudaStat2 = cudaMemcpy(LU, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
        cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
        assert(cudaSuccess == cudaStat3);

        if ( 0 > info ){
            printf("%d-th parameter is wrong \n", -info);
            exit(1);
        }
        if (pivot_on){
            printf("pivoting sequence, matlab base-1\n");
            for(int j = 0 ; j < m ; j++){
            printf("Ipiv(%d) = %d\n", j+1, Ipiv[j]);
            }
        }
    }

    /*
    * step 5: solve A*X = B
    */
    if (pivot_on){
        status = cusolverDnDgetrs(
                                cusolverH,
                                CUBLAS_OP_N,
                                m,
                                1,
                                d_A,
                                lda,
                                d_Ipiv,
                                d_B,
                                ldb,
                                d_info
                                );
    }
    else {
        status = cusolverDnDgetrs(
                                cusolverH,
                                CUBLAS_OP_N,
                                m,
                                1,
                                d_A,
                                lda,
                                NULL,
                                d_B,
                                ldb,
                                d_info
                            );
    }
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(X , d_B, sizeof(double) * m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //printf("solution:\nx[%d] = [", m);
    //for(int j = 0 ; j < m ; j++){
    //   printf(" %f ", X[j]);
    //}
    //printf("]'\n");

    if (doFactorization){
        printf("Elapsed Time (LU + Solve) for N = %d: %.7f ms\n", m, milliseconds);
    } else {
        printf("Elapsed Time (Solve only) for N = %d: %.7f ms\n", m, milliseconds);
    }

    /* free resources */
    if (d_A){
        cudaFree(d_A);
    }
    if (d_B){ 
        cudaFree(d_B);
    }
    if (d_Ipiv){
        cudaFree(d_Ipiv);
    }
    if (d_info){
        cudaFree(d_info);
    }
    if (d_work){
        cudaFree(d_work);
    }
    if (LU){
        free(LU);
    }
    if (Ipiv){
        free(Ipiv);
    }
    if (X){
        free(X);
    }
}

int main(int argc, char*argv[])
{
    int N = 10;
    int *n = (int*)malloc(N * sizeof(int));

    if (n == NULL) {
        fprintf(stderr, "Memory allocation failed for n.\n");
        exit(1);
    }

    for (int i = 0; i < N; i++){
        n[i] = pow(2, (i + 1));
    }

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;

    /* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    for (int i = 0; i < N; i++){
        double *H = HilbertMatrix(n[i]);
        double *b = B(n[i]);
        double *perturbedB = PerturbedB(n[i]);

        printf("\n--- Solving Hx = b for N = %d ---\n", n[i]);
        //PrintMatrix(H, n[i]);
        //printf("\n");
        //PrintB(b, n[i]);
        //printf("\n");
        HilbertLinearSystems(cusolverH, H, b, n[i], true);

        printf("\n--- Solving Hx = b_perturbed for N = %d ---\n", n[i]);
        //PrintMatrix(H, n[i]);
        //printf("\n");
        //PrintB(perturbedB, n[i]);
        //printf("\n");
        HilbertLinearSystems(cusolverH, H, perturbedB, n[i], false);

        printf("\n------------------------------------------------------------------------------------------------------------------\n");

        free(H);
        free(b);
        free(perturbedB);
    }

    if (cusolverH){
        cusolverDnDestroy(cusolverH);
    }
    if (stream){
        cudaStreamDestroy(stream);
    }
    if(n){
        free(n);
    }

    cudaDeviceReset(); // destroy context

    return 0;
}