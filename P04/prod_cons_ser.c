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
    for (int i = 0; i < N; i++) {
        sum += A[i];
    }
    return sum;
}

int main(){
    int N = 1000000000;  // Size of the array                                                                                                                                                               
    double *A, sum, runtime;
    int numthreads = 1;  // One thread at a time                                                                                                                                                            

    A = (double *)malloc(N * sizeof(double));
    if (A == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    omp_set_num_threads(numthreads);  // Set the number of threads to one                                                                                                                                   

    runtime = omp_get_wtime();

    // Thread 0 fills the array                                                                                                                                                                             
    fill_rand(N, A);

    // Then the same or another thread (still serially) sums the array                                                                                                                                      
    sum = Sum_Array(N, A);

    runtime = omp_get_wtime() - runtime;

    printf("\nWith %d thread and %lf seconds, The sum is %lf\n\n", numthreads, runtime, sum);

    free(A);  // Don't forget to free the allocated memory                                                                                                                                                  
    return 0;
}