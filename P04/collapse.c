#include <stdio.h>
#include <omp.h>

int main(void){
    #pragma omp parallel for collapse(2)
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                printf("Thread id %02d is responsible for (i=%02d,j=%02d)\n", omp_get_thread_num(), i, j);
            }
        }
    return 0;
}