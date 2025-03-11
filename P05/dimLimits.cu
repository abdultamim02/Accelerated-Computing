#include <stdio.h>

#define cudaCheckError(){ \
    cudaError_t e = cudaGetLastError(); \
    if(e != cudaSuccess){ \
        printf("Cuda Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void kernel(){
}

int main (int argc, char *argv[]){
    int blkx, blky, blkz, thdx, thdy, thdz;

    if (argc < 7) {
        printf("Usage: %s <blkx> <blky> <blkz> <thdx> <thdy> <thdz>\n", argv[0]);
        return 1;
    }

    sscanf(argv[1],"%d", &blkx);
    sscanf(argv[2],"%d", &blky);
    sscanf(argv[3],"%d", &blkz);
    sscanf(argv[4],"%d", &thdx);
    sscanf(argv[5],"%d", &thdy);
    sscanf(argv[6],"%d", &thdz);

    // dimension of block volume in grid
    dim3 dimGrid(blkx, blky, blkz);

    // dimension of thread volume per grid element (i.e., per block)
    // blockDim.x ≤ 1024, blockDim.y ≤ 1024, blockDim.z ≤ 64 But the total product must be ≤ --->1024<---
    dim3 dimBlock(thdx, thdy, thdz);

    kernel<<<dimGrid, dimBlock>>>();
    
    cudaCheckError();

    printf("Done\n");

    return 0;
}