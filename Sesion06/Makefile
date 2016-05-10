CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 -code=sm_35 -rdc=true -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc 
LD_FLAGS    = -lcudadevrt -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXESIM	        = kernelSIM.exe
EXEDYN	        = kernelDYN.exe
OBJSIM	        = mainSIM.o
OBJDYN	        = mainDYN.o


default: $(EXESIM)

mainSIM.o: mainSIM.cu
	$(NVCC) -c -o $@ mainSIM.cu $(NVCC_FLAGS)

mainDYN.o: mainDYN.cu
	$(NVCC) -c -o $@ mainDYN.cu $(NVCC_FLAGS)

$(EXESIM): $(OBJSIM)
	$(NVCC) $(OBJSIM) -o $(EXESIM) $(LD_FLAGS)

$(EXEDYN): $(OBJDYN)
	$(NVCC) $(OBJDYN) -o $(EXEDYN) $(LD_FLAGS)

all:	$(EXESIM) $(EXEDYN) 

clean:
	rm -rf *.o kernel*.exe

