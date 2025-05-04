#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>

#define ASSERT_NO_HIP_ERRORS(_hip_call)                                                      \
    do                                                                                       \
    {                                                                                        \
        auto _errcode = _hip_call;                                                           \
        if (_errcode != hipSuccess)                                                          \
        {                                                                                    \
            throw std::runtime_error(std::string("Hip Error: ") + std::to_string(_errcode)); \
        }                                                                                    \
    } while (false)

float rand01() { return (float)rand() / (float)RAND_MAX; }

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <density> <size>" << std::endl;
        return -1;
    }

    hipError_t error;
    float density = std::stof(argv[1]);
    const int size = std::stoi(argv[2]);

    std::vector<float> hx(size * size);
    std::vector<float> hy(size * size);
    std::vector<float> hC(size * size);
    std::vector<float> hC_blas(size * size);

    // generate random sparse data on host
    float f;

    for (int i = 0; i < size * size; i++)
    {
        f = rand01();
        // printf("%f\n",f);
        if (f < density)
        {
            hx[i] = f;
        }
        else
        {
            hx[i] = 0;
        }
    }

    for (int i = 0; i < size * size; i++)
    {
        f = rand01();
        // printf("%f\n",f);
        if (f < density)
        {
            hy[i] = f;
        }
        else
        {
            hy[i] = 0;
        }
    }

    // --- Allocate device memory ---
    float *dA, *dB, *dC;
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&dA, sizeof(float) * size * size));
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&dB, sizeof(float) * size * size));
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&dC, sizeof(float) * size * size));

    // --- Copy host matrices to device ---
    ASSERT_NO_HIP_ERRORS(hipMemcpy(dA, hx.data(), sizeof(float) * size * size, hipMemcpyHostToDevice));
    ASSERT_NO_HIP_ERRORS(hipMemcpy(dB, hy.data(), sizeof(float) * size * size, hipMemcpyHostToDevice));
    ASSERT_NO_HIP_ERRORS(hipMemset(dC, 0, sizeof(float) * size * size)); // clear C

    // compute host - side "exact" solution
    // hC = hx * hy
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            for (size_t k = 0; k < size; ++k)
            {
                hC[i * size + j] += hx[i * size + k] * hy[k * size + j];
            }
        }
    }

    printf("A = [\n");
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
            printf("%6.4f ", hx[i * size + j]);
        printf("\n");
    }
    printf("];\n");

    printf("B = [\n");
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
            printf("%6.4f ", hy[i * size + j]);
        printf("\n");
    }
    printf("];\n");

    printf("C = [\n");
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
            printf("%6.4f ", hC[i * size + j]);
        printf("\n");
    }
    printf("];\n");

    // --- Setup rocBLAS ---
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // --- Timing events ---
    hipEvent_t start, stop;
    ASSERT_NO_HIP_ERRORS(hipEventCreate(&start));
    ASSERT_NO_HIP_ERRORS(hipEventCreate(&stop));

    ASSERT_NO_HIP_ERRORS(hipEventRecord(start));

    // --- GEMM operation: C = alpha * A * B + beta * C ---
    rocblas_sgemm(handle,
                  rocblas_operation_transpose,
                  rocblas_operation_transpose,
                  size, size, size,
                  &alpha,
                  dA, size,
                  dB, size,
                  &beta,
                  dC, size);

    ASSERT_NO_HIP_ERRORS(hipEventRecord(stop));
    ASSERT_NO_HIP_ERRORS(hipEventSynchronize(stop));

    float milliseconds = 0;
    ASSERT_NO_HIP_ERRORS(hipEventElapsedTime(&milliseconds, start, stop));

    printf("rocBLAS: density %f, %f ms\n", density, milliseconds);

    ASSERT_NO_HIP_ERRORS(hipMemcpy(hC_blas.data(), dC, sizeof(float) * size * size, hipMemcpyDeviceToHost));

    if (size <= 32)
    {
        printf("hC_blas = [\n");
        for (size_t i = 0; i < size; ++i)
        {
            for (size_t j = 0; j < size; ++j)
            {
                printf("%6.4f ", hC_blas[j * size + i]);
            }
            printf("\n");
        }
        printf("];\n");
    }

    // --- Cleanup ---
    ASSERT_NO_HIP_ERRORS(hipFree(dA));
    ASSERT_NO_HIP_ERRORS(hipFree(dB));
    ASSERT_NO_HIP_ERRORS(hipFree(dC));
    rocblas_destroy_handle(handle);

    return 0;
}
