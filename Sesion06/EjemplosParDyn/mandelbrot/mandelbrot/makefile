CUDA_HOME   = /Soft/cuda/7.5.18
NVCC        = $(CUDA_HOME)/bin/nvcc

NAME=mandelbrot
SRC=$(NAME).cu
TGT=$(NAME)
TMP=*~ $(TGT)

build:	$(TGT)
$(TGT):	$(SRC) makefile
	$(NVCC) -O3 -arch=sm_20 -Xcompiler -fopenmp -lpng $(SRC) -o $(TGT)

run:	$(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
