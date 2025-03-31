
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main (int argc, char *argv[])
{
  #pragma omp parallel
  {
    printf("Hello from thread number %d\n", omp_get_thread_num());
  }
  return 0;
}
