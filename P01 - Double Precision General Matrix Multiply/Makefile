# Makefile Class Version

all: hi practice HW1_ser HW1_omp

#--------------------------------------------------------------------                                                                                                                                                                         
# Makefile for hi.c                                                                                                                                                                                                                           
hi: hi.o
        gcc -fopenmp hi.c -o hi
runHi: hi
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=4; ./hi)
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=8; ./hi)

#--------------------------------------------------------------------                                                                                                                                                                         
# Makefile for practice.c                                                                                                                                                                                                                     
practice: practice.o
        gcc -fopenmp practice.c -o practice
runPractice: practice
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=8; ./practice)

#--------------------------------------------------------------------                                                                                                                                                                         
# Makefile for HW1_ser.c                                                                                                                                                                                                                      
HW1_ser: HW1_ser.o
        gcc -fopenmp HW1_ser.c -o HW1_ser
runSer: HW1_ser
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=1; ./HW1_ser)

#--------------------------------------------------------------------                                                                                                                                                                         
# Makefile for HW1_omp.c                                                                                                                                                                                                                      
HW1_omp: HW1_omp.o
        gcc -fopenmp HW1_omp.c -o HW1_omp
runOmp: HW1_omp
        unset OMP_NUM_THREADS
        ./HW1_omp 1
        ./HW1_omp 1 | grep seconds
