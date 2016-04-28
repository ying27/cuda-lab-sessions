CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
PROG_FLAGS  = -DPINNED=1 -DDUMMY=30 -DTIMES=8

EXE00	        = kernel00.exe
EXE01	        = kernel01.exe
EXE02	        = kernel02.exe

OBJ00	        = stream00.o
OBJ01	        = stream01.o
OBJ02	        = stream02.o



default: $(EXE00)

stream00.o: stream00.cu
	$(NVCC) -c -o $@ stream00.cu $(NVCC_FLAGS) $(PROG_FLAGS)
stream01.o: stream01.cu
	$(NVCC) -c -o $@ stream01.cu $(NVCC_FLAGS) $(PROG_FLAGS)
stream02.o: stream02.cu
	$(NVCC) -c -o $@ stream02.cu $(NVCC_FLAGS) $(PROG_FLAGS)


$(EXE00): $(OBJ00)
	$(NVCC) $(OBJ00) -o $(EXE00) $(LD_FLAGS)

$(EXE01): $(OBJ01)
	$(NVCC) $(OBJ01) -o $(EXE01) $(LD_FLAGS)

$(EXE02): $(OBJ02)
	$(NVCC) $(OBJ02) -o $(EXE02) $(LD_FLAGS)


all:	$(EXE00) $(EXE01) $(EXE02) 

clean:
	rm -rf *.o kernel*.exe

