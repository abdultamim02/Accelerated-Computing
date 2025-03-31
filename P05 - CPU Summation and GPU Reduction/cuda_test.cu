#include <stdio.h>

__global__ void kernel() {
    printf("This is the CUDA Kernel\n");
    printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    kernel<<<2, 4>>>();  // 2 Blocks, 4 Threads each
    cudaDeviceSynchronize();
    return 0;
}
