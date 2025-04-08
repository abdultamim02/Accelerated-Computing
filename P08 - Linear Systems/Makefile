all: linear_systems hilbert_lineral_systems

#--------------------------------------------------------------------
# Makefile for lineral_systems.cu
linear_systems: linear_systems.cpp
	nvcc -c -I/usr/local/cuda/include linear_systems.cpp
	g++ -o linear_systems linear_systems.o -L/usr/local/cuda/lib64 -lcusolver -lcudart
run: linear_systems
	./linear_systems

all: linear_systems

#--------------------------------------------------------------------
# Makefile for hilbert_lineral_systems.cu
hilbert_lineral_systems: hilbert_lineral_systems.cpp
	nvcc -c -I/usr/local/cuda/include hilbert_lineral_systems.cpp
	g++ -o hilbert_lineral_systems hilbert_lineral_systems.o -L/usr/local/cuda/lib64 -lcusolver -lcudart
run: hilbert_lineral_systems
	./hilbert_lineral_systems