#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sched.h>
int sched_getcpu(void);

int main (int argc, char *argv[])
{
    #pragma omp parallel
    {
        printf("OMP thread %03d/%03d mapped to hwthread %03d\n",
        omp_get_thread_num(), omp_get_num_threads(), sched_getcpu());
    #pragma omp master
    {
        printf("This section is only executed once by thread %03d\n", omp_get_thread_num());
    }
    }
    return 0;
}