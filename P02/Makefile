# Makefile Class Version                                                                                                                                                                                                                      

all: HW2_ser HW2_omp

#--------------------------------------------------------------------                                                                                                                                                                         
# Makefile for HW2_ser.c                                                                                                                                                                                                                      
HW2_ser: HW2_ser.c
        gcc -fopenmp HW2_ser.c -o HW2_ser -lm
runSer: HW2_ser
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=1; ./HW2_ser)

#--------------------------------------------------------------------                                                                                                                                                                         
# Makefile for HW2_omp.c                                                                                                                                                                                                                      
HW2_omp: HW2_omp.c
        gcc -fopenmp HW2_omp.c -o HW2_omp -lm
runOmp1: HW2_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=1; ./HW2_omp 1)
runOmp2: HW2_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=2; ./HW2_omp 2)
runOmp4: HW2_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=4; ./HW2_omp 4)
runOmp8: HW2_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=8; ./HW2_omp 8)
runOmp16: HW2_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=16; ./HW2_omp 16)
runOmp32: HW2_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=32; ./HW2_omp 32)
runOmp64: HW2_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=64; ./HW2_omp 64)
runOmp128: HW2_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=128; ./HW2_omp 128)
runOmp256: HW2_omp
        unset OMP_NUM_THREADS
        export OMP_NUM_THREADS=256
        ./HW2_omp 256

#--------------------------------------------------------------------     
# Single Commands to Run HW2_omp.c with different numbers of threads and Nodes                                                                                                                                                                    
#       gcc -fopenmp HW2_omp.c -o HW2_omp -lm
#        unset OMP_NUM_THREADS
#        export OMP_NUM_THREADS=32
#        ./HW2_omp 32 1

#--------------------------------------------------------------------     
# Single Commands to Run HW2_ser.c with different numbers of Nodes 
#        gcc -fopenmp HW2_ser.c -o HW2_ser -lm
#        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=1; ./HW2_ser 1)