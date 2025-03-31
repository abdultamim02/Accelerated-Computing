all: convolution_2d_image

#--------------------------------------------------------------------
# Makefile for convolution_2d_image.cu
convolution_2d_image: convolution_2d_image.cu
	nvcc convolution_2d_image.cu -o convolution_2d_image
run: convolution_2d_image
	./convolution_2d_image