csrgemm.cc: 
	g++ -D__HIP_PLATFORM_AMD__  csrgemm.cc -I/opt/rocm-6.3.3/include -L/opt/rocm-6.3.3/lib -lamdhip64 -lrocblas -lrocsparse -o csrgemm
	./csrgemm 0.00001 8192
	./csrgemm 0.0001 8192
	./csrgemm 0.001 8192
	./csrgemm 0.01 8192
	./csrgemm 0.1 8192
	./csrgemm 1 8192

gemm.cc
	g++ -D__HIP_PLATFORM_AMD__  gemm.cc -I/opt/rocm-6.3.3/include -L/opt/rocm-6.3.3/lib -lamdhip64 -lrocblas -lrocsparse -o gemm
	./gemm 0.00001 8192
	./gemm 0.0001 8192
	./gemm 0.001 8192
	./gemm 0.01 8192
	./gemm 0.1 8192
	./gemm 1 8192

