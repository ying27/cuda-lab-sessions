CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

CONVEXE00		=	kernel00.exe
CONVOBJ00		=	conv00.o

CONVEXE01		=	kernel01.exe
CONVOBJ01		=	conv01.o



default: $(EXE)

conv00.o:	conv00.cu
	$(NVCC) -c -o $@ conv00.cu $(NVCC_FLAGS)

conv01.o:	conv01.cu
	$(NVCC) -c -o $@ conv01.cu $(NVCC_FLAGS)

$(CONVEXE00): $(CONVOBJ00)
	$(NVCC) $(CONVOBJ00) -o $(CONVEXE00) $(LD_FLAGS)

$(CONVEXE01): $(CONVOBJ01)
	$(NVCC) $(CONVOBJ01) -o $(CONVEXE01) $(LD_FLAGS)



clean:
	rm -rf *.o $(EXE)
