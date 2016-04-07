CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=sm_20 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE01	        = kernel01.exe
EXE02	        = kernel02.exe
EXE03	        = kernel03.exe
EXE04	        = kernel04.exe
EXE05	        = kernel05.exe
EXE06	        = kernel06.exe
EXE07	        = kernel07.exe
OBJ01	        = main01.o
OBJ02	        = main02.o
OBJ03	        = main03.o
OBJ04	        = main04.o
OBJ05	        = main05.o
OBJ06	        = main06.o
OBJ07	        = main07.o

default: $(EXE01)

main01.o: main01.cu
	$(NVCC) -c -o $@ main01.cu $(NVCC_FLAGS)

main02.o: main02.cu
	$(NVCC) -c -o $@ main02.cu $(NVCC_FLAGS)

main03.o: main03.cu
	$(NVCC) -c -o $@ main03.cu $(NVCC_FLAGS)

main04.o: main04.cu
	$(NVCC) -c -o $@ main04.cu $(NVCC_FLAGS)

main05.o: main05.cu
	$(NVCC) -c -o $@ main05.cu $(NVCC_FLAGS)

main06.o: main06.cu
	$(NVCC) -c -o $@ main06.cu $(NVCC_FLAGS)

main07.o: main07.cu
	$(NVCC) -c -o $@ main07.cu $(NVCC_FLAGS)

$(EXE01): $(OBJ01)
	$(NVCC) $(OBJ01) -o $(EXE01) $(LD_FLAGS)

$(EXE02): $(OBJ02)
	$(NVCC) $(OBJ02) -o $(EXE02) $(LD_FLAGS)

$(EXE03): $(OBJ03)
	$(NVCC) $(OBJ03) -o $(EXE03) $(LD_FLAGS)

$(EXE04): $(OBJ04)
	$(NVCC) $(OBJ04) -o $(EXE04) $(LD_FLAGS)

$(EXE05): $(OBJ05)
	$(NVCC) $(OBJ05) -o $(EXE05) $(LD_FLAGS)

$(EXE06): $(OBJ06)
	$(NVCC) $(OBJ06) -o $(EXE06) $(LD_FLAGS)

$(EXE07): $(OBJ07)
	$(NVCC) $(OBJ07) -o $(EXE07) $(LD_FLAGS)

all:	$(EXE01) $(EXE02) $(EXE03) $(EXE04) $(EXE05) $(EXE06) $(EXE07)

clean:
	rm -rf *.o kernel*.exe

