#include <stdio.h>
#define N 3

__global__ void kernel(float *d_A, float *d_B, float *d_C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n){
        d_C[i] = d_A[i] + d_B[i];
    }
}

void vecAdd(float *h_A, float *h_B, float *h_C, int n){
    // 0. Create device vectors A, B, and C
    float *d_A, *d_B, *d_C;
    
    // 1. Allocate device memory for vectors A, B, and C
    cudaMalloc((void **) &d_A, n * sizeof(float));
    cudaMalloc((void **) &d_B, n * sizeof(float));
    cudaMalloc((void **) &d_C, n * sizeof(float));

    // 2. Copy vectors A and B to device memory
    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Invoke kernel to perform vector addition
    kernel<<<1,3>>>(d_A, d_B, d_C,n);

    // 4. Copy vector C from the device memory to the host memory
    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Free device memory allocated for vectors
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main (int argc, char *argv[]){
    float h_A[N] = {1. ,2. ,3.};
    float h_B[N] = {4. ,5. ,6.};
    float h_C[N] = {0. ,0. ,0.};

    vecAdd(h_A, h_B, h_C, N);

    for (int i = 0; i < N; ++i)
        printf("%.1f ", h_C[i]);
        puts("");
        
    return 0;
}