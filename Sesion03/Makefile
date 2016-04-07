CUDA_HOME   = /Soft/cuda/7.5.18


NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
PROG_FLAGS  = -DPINNED=1

EXE00	        = kernel00.exe
EXE01	        = kernel01.exe
EXE02	        = kernel02.exe
EXE10	        = kernel10.exe
EXE11	        = kernel11.exe
EXE12	        = kernel12.exe

OBJ00	        = MM00.o
OBJ01	        = MM01.o
OBJ02	        = MM02.o
OBJ10	        = MM10.o
OBJ11	        = MM11.o
OBJ12	        = MM12.o

default: $(EXE00)

MM00.o: MM00.cu
	$(NVCC) -c -o $@ MM00.cu $(NVCC_FLAGS) $(PROG_FLAGS)
MM01.o: MM01.cu
	$(NVCC) -c -o $@ MM01.cu $(NVCC_FLAGS) $(PROG_FLAGS)
MM02.o: MM02.cu
	$(NVCC) -c -o $@ MM02.cu $(NVCC_FLAGS) $(PROG_FLAGS)
MM10.o: MM10.cu
	$(NVCC) -c -o $@ MM10.cu $(NVCC_FLAGS) $(PROG_FLAGS)
MM11.o: MM11.cu
	$(NVCC) -c -o $@ MM11.cu $(NVCC_FLAGS) $(PROG_FLAGS)
MM12.o: MM12.cu
	$(NVCC) -c -o $@ MM12.cu $(NVCC_FLAGS) $(PROG_FLAGS)


$(EXE00): $(OBJ00)
	$(NVCC) $(OBJ00) -o $(EXE00) $(LD_FLAGS)

$(EXE01): $(OBJ01)
	$(NVCC) $(OBJ01) -o $(EXE01) $(LD_FLAGS)

$(EXE02): $(OBJ02)
	$(NVCC) $(OBJ02) -o $(EXE02) $(LD_FLAGS)

$(EXE10): $(OBJ10)
	$(NVCC) $(OBJ10) -o $(EXE10) $(LD_FLAGS)

$(EXE11): $(OBJ11)
	$(NVCC) $(OBJ11) -o $(EXE11) $(LD_FLAGS)

$(EXE12): $(OBJ12)
	$(NVCC) $(OBJ12) -o $(EXE12) $(LD_FLAGS)


all:	$(EXE00) $(EXE01) $(EXE02) $(EXE10) $(EXE11) $(EXE12) 

clean:
	rm -rf *.o kernel*.exe

