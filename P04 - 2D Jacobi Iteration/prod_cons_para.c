#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to fill the array with random numbers                                                                                                                                                                                                                                                                                                                                                                                                                                           
void fill_rand(int N, double *A) {
    for (int i = 0; i < N; i++) {
        A[i] = (double)rand() / RAND_MAX;  // Random numbers between 0 and 1                                                                                                                                                                                                                                                                                                                                                                                                                
    }
}

// Function to sum the elements of the array                                                                                                                                                                                                                                                                                                                                                                                                                                                
double Sum_Array(int N, double *A) {
    double sum = 0.0;
    #pragma omp parallel            // Begins the parallel region                                                                                                                                                                                                                                                                                                                                                                                                                           
    {
        #pragma omp for reduction(+:sum)  // Apply the reduction to the for loop                                                                                                                                                                                                                                                                                                                                                                                                            
        for (int i = 0; i < N; i++) {
            sum += A[i];
        }
    }
    return sum;
}

int main(){
    int N = 1000000000;  // Size of the array                                                                                                                                                                                                                                                                                                                                                                                                                                               
    double *A, sum, runtime;
    int numthreads = 2;  // 2 threads                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    int flag = 0;

    A = (double *)malloc(N * sizeof(double));
    if (A == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    omp_set_num_threads(numthreads);
    runtime = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp master
        {
            numthreads = omp_get_num_threads();  // Get the number of threads                                                                                                                                                                                                                                                                                                                                                                                                               
            if (numthreads != 2)
            {
                printf("error: incorrect number of threads, %d \n", numthreads);
                exit(-1);
            }
        }
        #pragma omp barrier

        #pragma omp sections
        {
            #pragma omp section                 // First Section to fill A with random numbers                                                                                                                                                                                                                                                                                                                                                                                              
            {
                fill_rand(N, A);
                // #pragma omp flush below ensures that all modifications to the array A made by this thread are visible to other                                                                                                                                                                                                                                                                                                                                                           
                // threads. This is crucial for the second thread depend on the data in A being up-to-date and correctly initialized                                                                                                                                                                                                                                                                                                                                                        
                #pragma omp flush
                flag = 1;
                // #pragma omp flush (flag) ensures that the update to the variable flag is immediately visible to other threads.                                                                                                                                                                                                                                                                                                                                                           
                #pragma omp flush (flag)
            }
            #pragma omp section                 // Second Section to sum the elements in A                                                                                                                                                                                                                                                                                                                                                                                                  
            {
                // #pragma omp flush (flag) ensures that the current thread's view of flag is synchronized with the main memory.                                                                                                                                                                                                                                                                                                                                                            
                // This is critical just before entering a loop that depends on the value of flag.                                                                                                                                                                                                                                                                                                                                                                                          
                #pragma omp flush (flag)
                while (flag != 1){
                    // Continuously check the flag                                                                                                                                                                                                                                                                                                                                                                                                                                          
                    #pragma omp flush (flag)
                }
                // #pragma omp flush below ensure that all memory writes made by other threads, not just those to flag, are                                                                                                                                                                                                                                                                                                                                                                 
                // visible to the current thread                                                                                                                                                                                                                                                                                                                                                                                                                                            
                #pragma omp flush
                sum = Sum_Array(N, A);
            }
        }
    }

    runtime = omp_get_wtime() - runtime;

    printf("\nWith %d thread and %lf seconds, The sum is %lf\n\n", numthreads, runtime, sum);

    free(A);  // Free the allocated memory                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    return 0;
}
