#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>
#include <rocblas/rocblas.h>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <string>
#include <stdlib.h>
#include <math.h>

#define ASSERT_NO_HIP_ERRORS(_hip_call)                         \
    do                                                          \
    {                                                           \
        auto _errcode = _hip_call;                              \
        if (_errcode != hipSuccess)                             \
        {                                                       \
            throw std::runtime_error(std::string("Hip Error") + \
                                     std::to_string(_errcode)); \
        }                                                       \
    } while (false)

float rand01() { return (float)rand() / (float)RAND_MAX; }

void verify_rocSPARSE_output(const float *hC, const float *hValC, size_t SIZE, int size, size_t nnz_C)
{
    int OK = 1;

    for (size_t i = 0; i < SIZE && OK; ++i)
    {
        for (size_t j = 0; j < SIZE; ++j)
        {
            size_t idx = i * size + j;
            if (idx >= nnz_C)
                break;

            float cpu = hC[idx];
            float gpu = hValC[idx];

            if (fabsf(cpu - gpu) > 1e-2f)
            {
                printf("Mismatch at (%zu, %zu): CPU=%.4f, GPU=%.4f\n", i, j, cpu, gpu);
                OK = 0;
                break;
            }
        }
    }

    printf("rocSPARSE Verification: %s\n", (OK ? "PASSED" : "FAILED"));
}

void verify_rocBLAS_output(const float *hC_blas, const float *hC, int SIZE, int size)
{
    int ok = 1;
    for (int i = 0; i < SIZE && ok; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            float gpu = hC_blas[j * size + i]; // column-major → row-major
            float cpu = hC[i * size + j];
            if (fabsf(gpu - cpu) > 1e-2f)
            {
                printf("Mismatch @(%d,%d): GPU=%.4f CPU=%.4f\n", i, j, gpu, cpu);
                ok = 0;
                break;
            }
        }
    }
    printf("rocBLAS Verification: %s\n", (ok ? "PASSED" : "FAILED"));
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <density> <size>\n", argv[0]);
        return -1;
    }

    hipError_t error;
    char *string, *stopstring;

    float density = strtof(argv[1], &stopstring);
    const static int size = strtod(argv[2], &stopstring);

    std::vector<float> hA(size * size);
    std::vector<float> hB(size * size);
    std::vector<float> hC(size * size);
    std::vector<float> hC_blas(size * size);
    std::vector<float> hD(size * size);

    rocsparse_operation trans_A = rocsparse_operation_none;
    rocsparse_operation trans_B = rocsparse_operation_none;

    rocsparse_int nnz_A;
    rocsparse_int nnz_B;
    rocsparse_int nnz_C;
    rocsparse_int nnz_D;

    rocsparse_mat_descr descr_A;
    rocsparse_mat_descr descr_B;
    rocsparse_mat_descr descr_C;
    rocsparse_mat_descr descr_D;

    rocsparse_create_mat_descr(&descr_A);
    rocsparse_create_mat_descr(&descr_B);
    rocsparse_create_mat_descr(&descr_C);
    rocsparse_create_mat_descr(&descr_D);

    float *csr_val_A;
    float *csr_val_B;
    float *csr_val_C;
    float *csr_val_D;

    rocsparse_int *csr_row_ptr_A;
    rocsparse_int *csr_row_ptr_B;
    rocsparse_int *csr_row_ptr_C;
    rocsparse_int *csr_row_ptr_D;

    rocsparse_int *csr_col_ind_A;
    rocsparse_int *csr_col_ind_B;
    rocsparse_int *csr_col_ind_C;
    rocsparse_int *csr_col_ind_D;

    std::vector<int> hRowPtrA, hColIndexA;
    std::vector<float> hValA;

    std::vector<int> hRowPtrB, hColIndexB;
    std::vector<float> hValB;

    std::vector<int> hRowPtrC, hColIndexC;
    std::vector<float> hValC;

    std::vector<int> hRowPtrD, hColIndexD;
    std::vector<float> hValD;

    hipEvent_t start, stop;

    rocsparse_status status;

    error = hipEventCreate(&start);
    error = hipEventCreate(&stop);

    hRowPtrA.clear();
    hColIndexA.clear();
    hValA.clear();

    hRowPtrB.clear();
    hColIndexB.clear();
    hValB.clear();

    hRowPtrC.clear();
    hColIndexC.clear();
    hValC.clear();

    hRowPtrD.clear();
    hColIndexD.clear();
    hValD.clear();

    // generate random sparse data on host
    float f;

    for (int i = 0; i < size * size; i++)
    {
        f = rand01();
        // printf("%f\n",f);
        if (f < density)
        {
            hA[i] = f;
        }
        else
        {
            hA[i] = 0;
        }
    }

    for (int i = 0; i < size * size; i++)
    {
        f = rand01();
        // printf("%f\n",f);
        if (f < density)
        {
            hB[i] = f;
        }
        else
        {
            hB[i] = 0;
        }
    }

    // Convert to A and B to CSR format
    int ptr = 0;
    for (int i = 0; i < size; i++)
    {
        hRowPtrA.push_back(ptr);
        for (int j = 0; j < size; j++)
        {
            if (hA[i * size + j] != 0)
            {
                hValA.push_back(hA[i * size + j]);
                hColIndexA.push_back(j);
                ptr++;
            }
        }
    }
    hRowPtrA.push_back(ptr);
    nnz_A = ptr;

    ptr = 0;
    for (int i = 0; i < size; i++)
    {
        hRowPtrB.push_back(ptr);
        for (int j = 0; j < size; j++)
        {
            if (hB[i * size + j] != 0)
            {
                hValB.push_back(hB[i * size + j]);
                hColIndexB.push_back(j);
                ptr++;
            }
        }
    }
    hRowPtrB.push_back(ptr);
    nnz_B = ptr;

    ptr = 0;
    for (int i = 0; i < size; i++)
    {
        hRowPtrD.push_back(ptr);
    }
    hRowPtrD.push_back(ptr);
    nnz_D = ptr;

    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_val_A, sizeof(float) * nnz_A));
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_row_ptr_A, sizeof(float) * (size + 1)));
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_col_ind_A, sizeof(float) * nnz_A));

    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_val_B, sizeof(float) * nnz_B));
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_row_ptr_B, sizeof(float) * (size + 1)));
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_col_ind_B, sizeof(float) * nnz_B));

    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_val_D, sizeof(float) * nnz_D));
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_row_ptr_D, sizeof(float) * (size + 1)));
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_col_ind_D, sizeof(float) * nnz_D));

    ASSERT_NO_HIP_ERRORS(hipMemcpy(csr_val_A, hValA.data(), sizeof(float) * nnz_A, hipMemcpyHostToDevice));
    ASSERT_NO_HIP_ERRORS(hipMemcpy(csr_row_ptr_A, hRowPtrA.data(), sizeof(float) * (size + 1), hipMemcpyHostToDevice));
    ASSERT_NO_HIP_ERRORS(hipMemcpy(csr_col_ind_A, hColIndexA.data(), sizeof(float) * nnz_A, hipMemcpyHostToDevice));

    ASSERT_NO_HIP_ERRORS(hipMemcpy(csr_val_B, hValB.data(), sizeof(float) * nnz_B, hipMemcpyHostToDevice));
    ASSERT_NO_HIP_ERRORS(hipMemcpy(csr_row_ptr_B, hRowPtrB.data(), sizeof(float) * (size + 1), hipMemcpyHostToDevice));
    ASSERT_NO_HIP_ERRORS(hipMemcpy(csr_col_ind_B, hColIndexB.data(), sizeof(float) * nnz_B, hipMemcpyHostToDevice));

    ASSERT_NO_HIP_ERRORS(hipMemcpy(csr_val_D, hValD.data(), sizeof(float) * nnz_D, hipMemcpyHostToDevice));
    ASSERT_NO_HIP_ERRORS(hipMemcpy(csr_row_ptr_D, hRowPtrD.data(), sizeof(float) * (size + 1), hipMemcpyHostToDevice));
    ASSERT_NO_HIP_ERRORS(hipMemcpy(csr_col_ind_D, hColIndexD.data(), sizeof(float) * nnz_D, hipMemcpyHostToDevice));

    // compute host - side "exact" solution for small matrices
    // hC = hA * hB
    const size_t limit = std::min((size_t)4, (size_t)size); // Limit to first 4 rows and columns

for (size_t i = 0; i < limit; ++i)
{
    for (size_t j = 0; j < limit; ++j)
    {
        for (size_t k = 0; k < size; ++k)  // k loops full size
        {
            hC[i * size + j] += hA[i * size + k] * hB[k * size + j];
        }
    }
}


    int SIZE = std::min((size_t)4, (size_t)size);

    /*
    printf("A = [\n");
    for (size_t i = 0; i < SIZE; ++i)
    {
        for (size_t j = 0; j < SIZE; ++j)
        {
            printf("%6.4f ", hA[i * size + j]);
        }
        printf("\n");
    }
    printf("];\n");

    printf("B = [\n");
    for (size_t i = 0; i < SIZE; ++i)
    {
        for (size_t j = 0; j < SIZE; ++j)
        {
            printf("%6.4f ", hB[i * size + j]);
        }
        printf("\n");
    }
    printf("];\n");

    printf("C = [\n");
    for (size_t i = 0; i < SIZE; ++i)
    {
        for (size_t j = 0; j < SIZE; ++j)
        {
            printf("%6.4f ", hC[i * size + j]);
        }
        printf("\n");
    }
    printf("];\n");
    */

    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    float alpha = 1.0;
    float beta = 0.0;

    error = hipEventRecord(start);

    // Set pointer mode
    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);

    // Create matrix info structure
    rocsparse_mat_info info_C;
    rocsparse_create_mat_info(&info_C);

    // rocsparse_csrgemm_buffer_size()
    size_t buffer_size = 0;
    rocsparse_int m = size;
    rocsparse_int n = size;
    rocsparse_int k = size;

    // Query rocsparse for the required buffer size
    status = rocsparse_scsrgemm_buffer_size(handle,
                                            trans_A,
                                            trans_B,
                                            m,
                                            n,
                                            k,
                                            &alpha,
                                            descr_A,
                                            nnz_A,
                                            csr_row_ptr_A,
                                            csr_col_ind_A,
                                            descr_B,
                                            nnz_B,
                                            csr_row_ptr_B,
                                            csr_col_ind_B,
                                            &beta,
                                            descr_D,
                                            nnz_D,
                                            csr_row_ptr_D,
                                            csr_col_ind_D,
                                            info_C,
                                            &buffer_size);

    if (status != rocsparse_status_success)
    {
        fprintf(stderr, "rocsparse_scsrgemm_buffer_size: error (%d)\n", status);
        exit(2);
    }

    // Allocate buffer
    void *buffer;
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&buffer, buffer_size));

    // Obtain number of total non-zero entries in C and row pointers of C
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_row_ptr_C, sizeof(rocsparse_int) * (m + 1)));

    status = rocsparse_csrgemm_nnz(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   m,
                                   n,
                                   k,
                                   descr_A,
                                   nnz_A,
                                   csr_row_ptr_A,
                                   csr_col_ind_A,
                                   descr_B,
                                   nnz_B,
                                   csr_row_ptr_B,
                                   csr_col_ind_B,
                                   descr_D,
                                   nnz_D,
                                   csr_row_ptr_D,
                                   csr_col_ind_D,
                                   descr_C,
                                   csr_row_ptr_C,
                                   &nnz_C,
                                   info_C,
                                   buffer);

    if (status != rocsparse_status_success)
    {
        fprintf(stderr, "rocsparse_csrgemm_nnz: error (%d)\n", status);
        exit(2);
    }

    // Compute column indices and values of C
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_col_ind_C, sizeof(rocsparse_int) * nnz_C));
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&csr_val_C, sizeof(float) * nnz_C));

    status = rocsparse_scsrgemm(handle,
                                rocsparse_operation_none,
                                rocsparse_operation_none,
                                m,
                                n,
                                k,
                                &alpha,
                                descr_A,
                                nnz_A,
                                csr_val_A,
                                csr_row_ptr_A,
                                csr_col_ind_A,
                                descr_B,
                                nnz_B,
                                csr_val_B,
                                csr_row_ptr_B,
                                csr_col_ind_B,
                                &beta,
                                descr_D,
                                nnz_D,
                                csr_val_D,
                                csr_row_ptr_D,
                                csr_col_ind_D,
                                descr_C,
                                csr_val_C,
                                csr_row_ptr_C,
                                csr_col_ind_C,
                                info_C,
                                buffer);

    if (status != rocsparse_status_success)
    {
        fprintf(stderr, "rocsparse_scsrgemm: error (%d)\n", status);
    }

    error = hipEventRecord(stop);
    error = hipEventSynchronize(stop);

    float milliseconds = 0;
    error = hipEventElapsedTime(&milliseconds, start, stop);
    printf("rocSPARSE: density %f, %f ms\n", density, milliseconds);

    hValC.clear();

    for (int i = 0; i < nnz_C; i++)
    {
        hValC.push_back(0.0);
    }

    ASSERT_NO_HIP_ERRORS(hipMemcpy(hValC.data(), csr_val_C, sizeof(float) * nnz_C, hipMemcpyDeviceToHost));

    /*
    if (size <= 32)
    {
        printf("hValC = [\n");
        for (size_t i = 0; i < nnz_C; ++i)
        {
            printf("%6.4f ", hValC[i]);

            if ((i + 1) % 16 == 0)
            {
                printf("\n");
            }
        }
        printf("];\n");
    }
    */

    verify_rocSPARSE_output(hC.data(), hValC.data(), SIZE, size, nnz_C);

    // rocBLAS GEMM operation
    float *dA, *dB, *dC;
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&dA, sizeof(float) * size * size));
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&dB, sizeof(float) * size * size));
    ASSERT_NO_HIP_ERRORS(hipMalloc((void **)&dC, sizeof(float) * size * size));

    ASSERT_NO_HIP_ERRORS(hipMemcpy(dA, hA.data(), sizeof(float) * size * size, hipMemcpyHostToDevice));
    ASSERT_NO_HIP_ERRORS(hipMemcpy(dB, hB.data(), sizeof(float) * size * size, hipMemcpyHostToDevice));
    ASSERT_NO_HIP_ERRORS(hipMemset(dC, 0, sizeof(float) * size * size)); // clear C

    rocblas_handle handle_blas;
    rocblas_create_handle(&handle_blas);

    hipEvent_t start_blas, stop_blas;
    ASSERT_NO_HIP_ERRORS(hipEventCreate(&start_blas));
    ASSERT_NO_HIP_ERRORS(hipEventCreate(&stop_blas));

    ASSERT_NO_HIP_ERRORS(hipEventRecord(start_blas));

    // --- GEMM operation: C = alpha * A * B + beta * C ---
    rocblas_sgemm(handle_blas,
                  rocblas_operation_transpose,
                  rocblas_operation_transpose,
                  size, size, size,
                  &alpha,
                  dA, size,
                  dB, size,
                  &beta,
                  dC, size);

    ASSERT_NO_HIP_ERRORS(hipEventRecord(stop_blas));
    ASSERT_NO_HIP_ERRORS(hipEventSynchronize(stop_blas));

    float milliseconds_blas = 0;
    ASSERT_NO_HIP_ERRORS(hipEventElapsedTime(&milliseconds_blas, start_blas, stop_blas));

    printf("rocBLAS: density %f, %f ms\n", density, milliseconds_blas);

    ASSERT_NO_HIP_ERRORS(hipMemcpy(hC_blas.data(), dC, sizeof(float) * size * size, hipMemcpyDeviceToHost));

    /*
    printf("hC_blas = [\n");
    for (size_t i = 0; i < SIZE; ++i)
    {
        for (size_t j = 0; j < SIZE; ++j)
        {
            printf("%6.4f ", hC_blas[j * size + i]);
        }
        printf("\n");
    }
    printf("];\n");
    */

    verify_rocBLAS_output(hC_blas.data(), hC.data(), SIZE, size);

    // Free resources
    ASSERT_NO_HIP_ERRORS(hipFree(dA));
    ASSERT_NO_HIP_ERRORS(hipFree(dB));
    ASSERT_NO_HIP_ERRORS(hipFree(dC));

    rocblas_destroy_handle(handle_blas);

    error = hipFree(csr_val_A);
    error = hipFree(csr_row_ptr_A);
    error = hipFree(csr_col_ind_A);

    error = hipFree(csr_val_B);
    error = hipFree(csr_row_ptr_B);
    error = hipFree(csr_col_ind_B);

    ASSERT_NO_HIP_ERRORS(hipFree(buffer));
    ASSERT_NO_HIP_ERRORS(hipFree(csr_row_ptr_C));
    ASSERT_NO_HIP_ERRORS(hipFree(csr_val_C));
    ASSERT_NO_HIP_ERRORS(hipFree(csr_col_ind_C));

    return 0;
}