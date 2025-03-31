#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define TILE_WIDTH 16

__global__ void convolution(unsigned char *in, int *mask, unsigned char *out, int imageChannels, int maskwidth, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    int in_start_col = Col - (maskwidth / 2);
    int in_start_row = Row - (maskwidth / 2);

    if (Col < w && Row < h) {
        for(int c = 0; c < imageChannels; c++){
            int pixVal = 0;
            for(int j = 0; j < maskwidth; ++j) {
                for(int k = 0; k < maskwidth; ++k) {
                    int curRow = in_start_row + j;
                    int curCol = in_start_col + k;
    
                    // Verify we have a valid image pixel
                    if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                        pixVal += in[(curRow * w + curCol) * imageChannels + c] * mask[j * maskwidth + k];
                    }
                }
            }
            // Write our new pixel value out (Clamp the result to [0, 255] to avoid any overflow or underflow)
            pixVal = min(max(pixVal, 0), 255);
            out[(Row * w + Col) * imageChannels + c] = (unsigned char)(pixVal);
        }
    }
}

int main(int argc, char *argv[])
{
    unsigned char *hostInputImage;
    unsigned char *hostOutputImage;
    unsigned int inputLength = 589824; // 384 * 512 * 3 = 589824

    printf("%% Importing 3-channel image data and creating memory on host\n");

    // Dynamic Array Alocation on Host (CPU) side
    hostInputImage = (unsigned char *)malloc(inputLength * sizeof(unsigned char));
    hostOutputImage = (unsigned char *)malloc(inputLength * sizeof(unsigned char));

    FILE *f;
    unsigned int pixelValue, i = 0;

    f = fopen("peppers.dat","r");

    while(!feof(f) && i < inputLength){
        fscanf(f, "%d", &pixelValue);
        hostInputImage[i++] = pixelValue;
    }

    fclose(f);

    int maskRows = 5;
    int maskColumns = 5;
    int imageChannels = 3;
    int imageWidth = 512;
    int imageHeight = 384;

    // Sobel 5x5 horizontal convolution kernel for edge detection
    int hostMask[5][5] = {
                        {2,   2,  4,  2,  2},
                        {1,   1,  2,  1,  1},
                        {0,   0,  0,  0,  0},
                        {-1, -1, -2, -1, -1},
                        {-2, -2, -4, -2, -2}
                        };

    unsigned char *deviceInputImage;
    unsigned char *deviceOutputImage;
    int *deviceMask;

    assert(maskRows == 5);
    assert(maskColumns == 5);

    // Dynamic Array Alocation on Device (GPU) side
    cudaMalloc((void **)&deviceInputImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    cudaMalloc((void **)&deviceOutputImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    cudaMalloc((void **)&deviceMask, maskRows * maskColumns * sizeof(int));

    cudaMemcpy(deviceInputImage, hostInputImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMask, hostMask, maskRows * maskColumns * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((float)imageWidth / TILE_WIDTH), ceil((float)imageHeight / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Launch kernel
    convolution<<<dimGrid, dimBlock>>>(deviceInputImage, deviceMask, deviceOutputImage, imageChannels, maskRows, imageWidth, imageHeight);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results from device to host
    cudaMemcpy(hostOutputImage, deviceOutputImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    f = fopen("peppers.out","w");
    for(int i = 0; i < inputLength; ++i){
        fprintf(f, "%d\n", hostOutputImage[i]);
    }
    fclose(f);

    // Free Allocated Memories
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);
    cudaFree(deviceMask);

    free(hostInputImage);
    free(hostOutputImage);
    
    return(0);
}