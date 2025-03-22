#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 512

__global__ void total(float *input, float *output, int size){
    // Load a segment of the input vector into shared memory
    __shared__ float partialSum [2 * BLOCK_SIZE];

    unsigned int t = threadIdx.x;                           // Represent a thread
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    if (start + t < size){
        partialSum[t] = input[start + t];
    }
    else{
        partialSum[t] = 0.0f;
    }

    if (start + BLOCK_SIZE + t < size){
        partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    }
    else{
        partialSum[BLOCK_SIZE + t] = 0.0f;
    }

    // Traverse the reduction tree
    /*
        strides will assume values:
        512
        256
        128
        64
        32
        16
        8
        4
        2
        1
    */

    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1){
        __syncthreads();

        if (t < stride){
            partialSum[t] += partialSum[t + stride];
        }
    }

    // Write the computed sum of the block to the output vector at the correct index
    if (t == 0){
        output[blockIdx.x] = partialSum[0];
    }
}

// SumSequentially function add N sequential floats, on the host (CPU) sequentially
double SumSequentially(int N){
    double result = 0.0;

    for(unsigned int i = 1; i <= N; i++){
        result += (double)i;         // Add each sequential float to the result
    }
    return result;
}

int main (int argc, char *argv[]){
    float *hostInput, *hostOutput, *deviceInput, *deviceOutput;
    int numInputElements, numOutputElements;
    double hostResult = 0.0, deviceResult = 0.0;

    clock_t t;

    cudaEvent_t start, stop;
    float elapsedTime;

    if (argc < 2) {
        printf("Usage: %s <numInputElements>\n", argv[0]);
        return 1;
    }

    sscanf(argv[1],"%d", &numInputElements);

    printf("%d elements\n", numInputElements);
    printf("Creating random input data on host\n");
    printf("numInputElements: %d\n", numInputElements);

    hostInput = (float *)malloc(numInputElements * sizeof(float));
    for (size_t i = 0; i < numInputElements; i++){
        hostInput[i] = (float) i + 1;
    }

    cudaMalloc(&deviceInput, sizeof(float) * numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE << 1);

    printf("numOutputElements: %d\n", numOutputElements);
    printf("BLOCK_SIZE: %d\n", BLOCK_SIZE);
    printf("Each thread block takes 2*BlockDim.x = %d input elements\n", 2 * BLOCK_SIZE);
    printf("adjusted numOutputElements: %d\n", numOutputElements);

    hostOutput = (float *)malloc(numOutputElements * sizeof(float));
    cudaMalloc(&deviceOutput, sizeof(float) * numOutputElements);

    printf("Allocating GPU memory\n");
    cudaMemcpy(deviceInput, hostInput, sizeof(float) * numInputElements, cudaMemcpyHostToDevice);
    printf("Copying input memory to the GPU\n");

    dim3 dimGrid(numOutputElements, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    printf("Performing CUDA computation\n");
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    // Device (GPU) Computation
    total<<<dimGrid, dimBlock>>> (deviceInput, deviceOutput, numInputElements);
    cudaDeviceSynchronize();

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("device elapsed time: %f ms\n", elapsedTime);

    printf("Copying output memory to the CPU\n");
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * numOutputElements, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numOutputElements; i++) {
        deviceResult += (double)hostOutput[i];
    }

    printf("Sum via GPU reduction: %.0f\n", deviceResult);

    t = clock();
    // Host (CPU) Computation
    hostResult = SumSequentially(numInputElements);
    t = clock() - t;

    printf("host elapsed time: %f ms\n" , ((double)t) / CLOCKS_PER_SEC * 1000);
    printf("Sum via host iteration: %.0f\n", hostResult);

    printf("Speedup (CPU Time / GPU Time): %f\n", (((double)t) / CLOCKS_PER_SEC * 1000) / elapsedTime);

    printf("Free GPU Memory\n");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    free(hostOutput);
    free(hostInput);

    return 0;
}
