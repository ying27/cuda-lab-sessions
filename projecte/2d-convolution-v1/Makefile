CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE	        = kernel.exe
OBJ	        = main.o

CONVEXE		=	conv.exe
CONVOBJ		=	conv.o




default: $(EXE)

main.o: main.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

conv.o:	conv.cu
	$(NVCC) -c -o $@ conv.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

$(CONVEXE): $(CONVOBJ)
	$(NVCC) $(CONVOBJ) -o $(CONVEXE) $(LD_FLAGS)




clean:
	rm -rf *.o $(EXE) $(CONVEXE)
