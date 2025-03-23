#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>

#define cudaCheckError(){ \
    cudaError_t e = cudaGetLastError(); \
    if(e != cudaSuccess){ \
        printf("Cuda Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

#define TILE_WIDTH 16

__global__ void matrixMultiply(float* A, float* B, float* C, int m, int p, int n)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < m) && (Col < n)) {
        float Cvalue = 0.0f;

        // Compute the matrix multiplication (dot product) of row or matrix A and column or matrix B
        for (int k = 0; k < p; ++k) {
            Cvalue += A[Row * p + k] * B[k * n + Col];
        }

        C[Row * n + Col] = Cvalue;
    }
}

__global__ void matrixTiledMultiply(float* A, float* B, float* C, int m, int p, int n)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    for (int k = 0; k < (p + TILE_WIDTH - 1) / TILE_WIDTH; ++k) {
        // Load tile from A
        if (Row < m && (k * TILE_WIDTH + tx) < p){
            tileA[ty][tx] = A[Row * p + k * TILE_WIDTH + tx];
        }
        else {
            tileA[ty][tx] = 0.0f;
        }

        // Load tile from B
        if ((k * TILE_WIDTH + ty) < p && Col < n){
            tileB[ty][tx] = B[(k * TILE_WIDTH + ty) * n + Col];
        }
        else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply tiles
        for (int i = 0; i < TILE_WIDTH; ++i){
            Cvalue += tileA[ty][i] * tileB[i][tx];
        }

        __syncthreads();
    }

    // Write the result
    if (Row < m && Col < n){
        C[Row * n + Col] = Cvalue;
    }
}


void matrixHostMultiply(float* A, float* B, float* C, int m, int p, int n){
    // Loop through rows of A
    for(int i = 0; i < m; i++){
        // Loop through columns of B
        for(int j = 0; j < n; j++){
            float value = 0.0f;

            // Multiply row of A with column of B
            for(int k = 0; k < p; k++){
                // A[i][k] * B[k][j]
                value += A[i * p + k] * B[k * n + j];
            }
            // Store result in C[i][j]
            C[i * n + j] = value;
        }
    }
}

void printMatrix(const float* C, int m, int n, const char* name) {
    printf("\nMatrix %s (%d x %d):\n", name, m, n);
    printf("C = [\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%8.2f ", C[i * n + j]);
        }
        printf("\n");
    }
    printf("]\n");
}

int main (int argc, char *argv[]){
    float *hA, *hB, *hC_global, *hC_tiled, *hC_host, *dA, *dB, *dC_global, *dC_tiled;
    int n, m, p;

    if (argc < 2) {
        printf("Usage: %s <m> <p> <n>\n", argv[0]);
        return 1;
    }

    sscanf(argv[1],"%d", &m);
    sscanf(argv[2],"%d", &p);
    sscanf(argv[3],"%d", &n);

    printf("Matrix A: %d x %d (m x p)\n", m, p);
    printf("Matrix B: %d x %d (p x n)\n", p, n);
    printf("Result C: %d x %d (m x n)\n", m, n);

    // Dynamic Array Alocation on Host (CPU) side
    hA = (float *)malloc(m * p * sizeof(float));            // Matirx A of dimension m x p
    hB = (float *)malloc(p * n * sizeof(float));            // Matrix B of dimension p x n
    hC_global = (float *)malloc(m * n * sizeof(float));     // Matrix C (Host for GPU Global kernel) of dimension m x n
    hC_tiled = (float *)malloc(m * n * sizeof(float));      // Matrix C (Host for GPU Tiled kernel) of dimension m x n
    hC_host = (float *)malloc(m * n * sizeof(float));       // Matrix C (Host for CPU) of dimension m x n

    // Seed random number generator
    srand(time(NULL));

    // Initialize Matrix A random values
    for (size_t i = 0; i < m * p; i++){
        hA[i] = ((float) rand() / (float) RAND_MAX) * 10.0f;    // random float 0.0 – 10.0
    }

    // Initialize Matrix B random values
    for (size_t i = 0; i < p * n; i++){
        hB[i] = ((float) rand() / (float) RAND_MAX) * 10.0f; // random float 0.0 – 10.0
    }

    // Dynamic Array Alocation on Device (GPU) side
    cudaMalloc(&dA, sizeof(float) * m * p);             // Matirx A of dimension m x p
    cudaMalloc(&dB, sizeof(float) * p * n);             // Matrix B of dimension p x n
    cudaMalloc(&dC_global, sizeof(float) * m * n);      // Matrix C (Host for GPU Global kernel) of dimension m x n
    cudaMalloc(&dC_tiled, sizeof(float) * m * n);       // Matrix C (Host for GPU Tiled kernel) of dimension m x n

    // Copy host memory to device
    cudaMemcpy(dA, hA, sizeof(float) * m * p, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float) * p * n, cudaMemcpyHostToDevice);

    // Initialize thread block and kernel grid dimensions
    dim3 dimGrid((n - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Compute the elapsed time and invoke CUDA matrixMultiply kernel (G: Global kernel)
    cudaEvent_t startG, stopG;

    cudaEventCreate(&startG);
    cudaEventCreate(&stopG);

    cudaEventRecord(startG);

    // Launch kernel
    matrixMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC_global, m, p, n);

    // Check for kernel launch errors
    cudaCheckError();

    cudaEventRecord(stopG);

    cudaEventSynchronize(stopG);

    float millisecondsG = 0;

    cudaEventElapsedTime(&millisecondsG, startG, stopG);

    cudaEventDestroy(startG);
    cudaEventDestroy(stopG);

    printf("matrixMultiply kernel (Global) elapsed time: %f ms\n", millisecondsG);

    // Compute the elapsed time and invoke CUDA matrixTiledMultiply kernel (T: Tiled kernel)
    cudaEvent_t startT, stopT;

    cudaEventCreate(&startT);
    cudaEventCreate(&stopT);

    cudaEventRecord(startT);

    // Launch kernel
    matrixTiledMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC_tiled, m, p, n);

    // Check for kernel launch errors
    cudaCheckError();

    cudaEventRecord(stopT);

    cudaEventSynchronize(stopT);

    float millisecondsT = 0;

    cudaEventElapsedTime(&millisecondsT, startT, stopT);

    cudaEventDestroy(startT);
    cudaEventDestroy(stopT);

    printf("matrixTiledMultiply kernel (Tiled) elapsed time: %f ms\n", millisecondsT);

    // Host (CPU) Computation
    //matrixHostMultiply(hA, hB, hC_host, m, p, n);

    // Copy results from device to host
    cudaDeviceSynchronize();
    cudaMemcpy(hC_global, dC_global, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hC_tiled, dC_tiled, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    //printMatrix(hC_host, m, n, "Host C");
    //printMatrix(hC_global, m, n, "Device C");
    //printMatrix(hC_tiled, m, n, "Tiled C");

    // Free Allocated Memories
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC_global);
    cudaFree(dC_tiled);

    free(hA);
    free(hB);
    free(hC_global);
    free(hC_tiled);
    free(hC_host);

    return 0;
}