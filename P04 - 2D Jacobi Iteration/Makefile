all: prod_cons_ser prod_cons_para jacobi_serial jacobi_omp jacobi_omp_gpu jacobi_omp_gpu_simd

#--------------------------------------------------------------------                                                                                         
# Makefile for prod_cons_ser.c                                                                                                                                
prod_cons_ser: prod_cons_ser.o
        gcc -fopenmp -c -o prod_cons_ser.o prod_cons_ser.c
        gcc -fopenmp prod_cons_ser.o -o prod_cons_ser
runSer: prod_cons_ser
        ./prod_cons_ser

#--------------------------------------------------------------------                                                                                         
# Makefile for prod_cons_para.c                                                                                                                               
prod_cons_para: prod_cons_para.o
        gcc -fopenmp -c -o prod_cons_para.o prod_cons_para.c
        gcc -fopenmp prod_cons_para.o -o prod_cons_para
runPar: prod_cons_para
        (unset OMP_NUM_THREADS; export0jRc!MCqKM OMP_NUM_THREADS=2; ./prod_cons_para)

#--------------------------------------------------------------------                                                                                         
# Makefile for collapse.c                                                                                                                                     
collapse: collapse.o
        gcc -fopenmp collapse.c -o collapse
runCol: collapse
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=32; ./collapse |sort -n -k2)

#--------------------------------------------------------------------                                                                                         
# Makefile for jacobi_serial.c jacobi_omp.c jacobi_omp_gpu.c and jacobi_omp_gpu_simd.c
CC = nvc
CFLAGS = -mp=gpu -gpu=cc80
jacobi_serial: jacobi_serial.c
        $(CC) $(CFLAGS) $(LIBS) $< -o ${@}.exe

jacobi_omp: jacobi_omp.c
        $(CC) $(CFLAGS) $(LIBS) $< -o ${@}.exe

jacobi_omp_gpu: jacobi_omp_gpu.c
        $(CC) $(CFLAGS) $(LIBS) $< -o ${@}.exe

jacobi_omp_gpu_simd: jacobi_omp_gpu_simd.c
        $(CC) $(CFLAGS) $(LIBS) $< -o ${@}.exe
        