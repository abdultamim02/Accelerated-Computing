all: reduction_sum_N rectangle_method_omp Simpson

#--------------------------------------------------------------------                                                                                                                                       
# Makefile for reduction_sum_N.cpp                                                                                                                                                                          
reduction_sum_N.o: reduction_sum_N.cpp
        g++ -fopenmp -c reduction_sum_N.cpp -o reduction_sum_N.o
reduction_sum_N: reduction_sum_N.o
        g++ -fopenmp reduction_sum_N.o -o reduction_sum_N
run: reduction_sum_N
        export OMP_PLACES=threads; \
        export OMP_PROC_BIND=true; \
        export OMP_NUM_THREADS=128; \
        i=1; \
        while [ $$i -lt 10 ]; do \
                value=$$(echo "10^$$i" | bc) ; \
                echo "Processing $$value or 1e-$$i" ; \
                ./reduction_sum_N $$value ; \
                i=$$(($$i + 1)); \
                echo "" ;\
        done

#--------------------------------------------------------------------                                                                                                                                       
# Makefile for rectangle_method_omp.c                                                                                                                                                                       
rectangle_method_omp: rectangle_method_omp.o
        gcc -fopenmp rectangle_method_omp.c -o rectangle_method_omp -lm
run160: rectangle_method_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=16; ./rectangle_method_omp 1.0 2.0 160 16)
run1600: rectangle_method_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=16; ./rectangle_method_omp 1.0 2.0 1600 16)
run16000: rectangle_method_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=16; ./rectangle_method_omp 1.0 2.0 16000 16)
run160000: rectangle_method_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=16; ./rectangle_method_omp 1.0 2.0 160000 16)
run1600000: rectangle_method_omp
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=16; ./rectangle_method_omp 1.0 2.0 1600000 16)

#--------------------------------------------------------------------                                                                                                             
# Makefile for Simpson.cpp                                                                                                                                                        
Simpson: Simpson.o
        g++ -fopenmp Simpson.cpp -o Simpson
run1: Simpson
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=1; ./Simpson 1 1e6)
run2: Simpson
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=2; ./Simpson 2 1e6)
run4: Simpson
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=4; ./Simpson 4 1e6)
run8: Simpson
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=8; ./Simpson 8 1e6)
run16: Simpson
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=16; ./Simpson 16 1e6)
run32: Simpson
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=32; ./Simpson 32 1e6)
run64: Simpson
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=64; ./Simpson 64 1e6)
run128: Simpson
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=128; ./Simpson 128 1e6)
run256: Simpson
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=256; ./Simpson 256 1e6)

g++ -fopenmp Simpson.cpp -o Simpson
(unset OMP_NUM_THREADS; export OMP_NUM_THREADS=16; ./Simpson 16 1e6)