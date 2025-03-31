#include <stdio.h>
#define N 3

void vecAdd(float *h_A, float *h_B, float *h_C, int n){
    int i;
    for (i = 0; i < n; i++){
        h_C[i] = h_A[i] + h_B[i];
    }
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