#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "PNG.h"

#define NGPUS 4

#ifndef KSIZE
#define KSIZE 9
#endif


#ifndef SIZE
#define SIZE 32
#endif


void print_matrix(unsigned char* matrix, int size, int num_comp, int w) {
	for (int i = 1; i <= size; i++) {
		printf("%d ", matrix[i]);
		if (i%num_comp == 0) {
			printf("| ");
			if (i%w == 0) printf("\n");
		}
	}
	printf("\n\n");
} 



//N: numero de columnes en components
//M: numero de files

//N && M correspon al tamany de la matriu de output
__global__ void convolution (const int head, const int tail, const int N, const int M, unsigned char* input, unsigned char* output){

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

	int row_in = row + head;
        
	if (row < M && col < N) {
		output[row*N+col] = input[row_in*N+col];
	}
}



int main(int argc, char** argv)
{
	//PNG inPng("blanc_10_10.png");
	PNG inPng("pixar.png");
	PNG outPng;
	outPng.Create(inPng.w, inPng.h);

        const unsigned int w = inPng.w; 
        const unsigned int h = inPng.h;
        //const unsigned int N = (w > h) ? w : h;
        int size = w * h;
        int size4 = size*4;
	int k2 = KSIZE/2;

	
	//unsigned int nBlocks, nThreads;
	//nThreads = SIZE;
	//nBlocks = N/nThreads + (N%nThreads != 0);
	//dim3 dimGrid(nBlocks, nBlocks, 1);
	//dim3 dimBlock(nThreads, nThreads, 1);

	//posicio [0]: Head / [1]: numero bytes / [2] = head + offset / [3] = bytes + offset
	int dimensions[NGPUS][4];
	int k = (inPng.h/NGPUS);
	int rest = inPng.h-(k*NGPUS);
	int i = 0;

	for (int j = 0; j < size4;) {
		dimensions[i][0] = j;
		dimensions[i][1] = (k + (i < rest))*inPng.w*4;
		dimensions[i][2] = dimensions[i][0] - k2*(j != 0)*inPng.w*4;
		dimensions[i][3] = dimensions[i][1] + (k2*(i != (NGPUS-1)) + k2*(j != 0))*inPng.w*4;

		j = j + (k + (i < rest))*inPng.w*4;
		i++;
	}
	
	/*Testing
	printf("Height: %d, Width: %d, size: %d, chunk: %d\n", inPng.h, inPng.w*4, size4, k*4*inPng.w);
	
	for (int j = 0;j < NGPUS; j++){
		printf("GPU-%d: head = %d / tail = %d / chunk = %d / head+offset = %d / bytes+offset = %d\n", 
		      j, dimensions[j][0], dimensions[j][0] + dimensions[j][1], dimensions[j][1], dimensions[j][2], dimensions[j][3]);
	}
	*/
	
	
	char *d_input[NGPUS];
	char *d_output[NGPUS];
	char *h_output;

	cudaMallocHost((float**)&h_output,  size4*sizeof(unsigned char));

        //Allocate memory in the devices
        for (int i = 0; i < NGPUS; i++){
                //Set the device
                cudaSetDevice(i);

                //Allocate memory in the device for the base image and the single component image
                cudaMalloc((void**)&d_input[i], dimensions[i][3] * sizeof(unsigned char));
		cudaMalloc((void**)&d_output[i], dimensions[i][1] * sizeof(unsigned char));

                //Copy asynchronously the base image to the device
                cudaMemcpyAsync(d_input[i], &inPng.data[dimensions[i][2]], dimensions[i][3] * sizeof(unsigned char), cudaMemcpyHostToDevice);
        }

	int krest = k + (rest != 0);
	const unsigned int N = (w > krest) ? w*4 : krest*4;
        unsigned int nBlocks, nThreads;
        nThreads = SIZE;
        nBlocks = N/nThreads + (N%nThreads != 0);
        dim3 dimGrid(nBlocks, nBlocks, 1);
        dim3 dimBlock(nThreads, nThreads, 1);


	
	for (int i = 0; i < NGPUS; i++){
                //Set the device
                cudaSetDevice(i);
		convolution<<<dimGrid,dimBlock>>> ((i != 0)*k2, (i != (NGPUS-1))*k2, w*4, dimensions[i][1]/(w*4), (unsigned char*)d_input[i], (unsigned char*)d_output[i]);
        }

	
	
	for (int i = 0; i < NGPUS; i++) {
		cudaSetDevice(i);
		cudaMemcpyAsync(&h_output[dimensions[i][0]], d_output[i], dimensions[i][1], cudaMemcpyDeviceToHost);
	}



        cudaDeviceSynchronize();
        std::copy(&h_output[0], &h_output[size4], std::back_inserter(outPng.data));
        outPng.Save("result.png");
        print_matrix(&outPng.data[0], size4, 4, inPng.w*4);
	
	return 0;

}
