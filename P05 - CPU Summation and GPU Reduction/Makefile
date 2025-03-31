all: cude_kernel cude_test cuda_malloc_man VecAddThrust VecAddHost VecAddDevice dimLimits gpu_parallel_reduction

#--------------------------------------------------------------------                                                                                         
# Makefile for cuda_kernel.cu
cuda_kernel: cuda_kernel.cu
	nvcc cuda_kernel.cu -o cuda_kernel
runKernel: cuda_kernel
	./cuda_kernel

#--------------------------------------------------------------------                                                                                         
# Makefile for cuda_test.cu
cuda_test: cuda_test.cu
	nvcc cuda_test.cu -o cuda_test
runTest: cuda_test
	./cuda_test

#--------------------------------------------------------------------                                                                                         
# Makefile for cuda_malloc_man.cu
cuda_malloc_man: cuda_malloc_man.cu
	nvcc cuda_malloc_man.cu -o cuda_malloc_man
runMalloc: cuda_malloc_man
	./cuda_malloc_man

#--------------------------------------------------------------------                                                                                         
# Makefile for VecAddThrust.cu
VecAddThrust: VecAddThrust.cu
	nvcc VecAddThrust.cu -o VecAddThrust
runVec: VecAddThrust
	./VecAddThrust

#--------------------------------------------------------------------                                                                                         
# Makefile for VecAddHost.cu
VecAddHost: VecAddHost.cu
	nvcc VecAddHost.cu -o VecAddHost
runVecH: VecAddHost
	./VecAddHost

#--------------------------------------------------------------------                                                                                         
# Makefile for VecAddDevice.cu
VecAddDevice: VecAddDevice.cu
	nvcc VecAddDevice.cu -o VecAddDevice
runVecD: VecAddDevice
	./VecAddDevice

#--------------------------------------------------------------------                                                                                         
# Makefile for dimLimits.cu
dimLimits: dimLimits.cu
	nvcc dimLimits.cu -o dimLimits
runDim1: dimLimits
	./dimLimits 2147483647 65535 65535 2 2 2
runDim2: dimLimits
	./dimLimits 2147483648 65535 65535 2 2 2
runDim3: dimLimits
	./dimLimits 2147483647 65535 65535 1024 2 2
runDim4: dimLimits
	./dimLimits 2147483647 65535 65535 1025 2 2
runDim5: dimLimits
	./dimLimits 2147483647 65535 65535 1024 1024 64
runDim6: dimLimits
	./dimLimits 2147483647 65535 65535 1024 1024 65

#--------------------------------------------------------------------                                                                                         
# Makefile for gpu_parallel_reduction.cu
gpu_parallel_reduction: gpu_parallel_reduction.cu
	nvcc gpu_parallel_reduction.cu -o gpu_parallel_reduction
runReduction1: gpu_parallel_reduction
	./gpu_parallel_reduction 1024
	./gpu_parallel_reduction 2048
	./gpu_parallel_reduction 4096
	./gpu_parallel_reduction 8192
	./gpu_parallel_reduction 16384
	./gpu_parallel_reduction 32768
	./gpu_parallel_reduction 65536
	./gpu_parallel_reduction 131072
	./gpu_parallel_reduction 262144
	./gpu_parallel_reduction 524288
	./gpu_parallel_reduction 1048576
	./gpu_parallel_reduction 1073741824
	