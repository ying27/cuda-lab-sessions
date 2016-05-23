#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "PNG.h"

#define NGPUS 4

#ifndef KSIZE
#define KSIZE 3
#endif


#ifndef SIZE
#define SIZE 32
#endif


void print_matrix(unsigned char* matrix, int size, int num_comp, int w) {
	for (int i = 1; i <= size; i++) {
		printf("%d ", matrix[i]);
		if (i%num_comp == 0) {
			printf("| ");
			if (i%w == 0) printf("\n\n");
		}
	}
	printf("\n");
} 



//N: numero de columnes en components
//M: numero de files

//N && M correspon al tamany de la matriu de output
__global__ void convolution (int head, int N, int M, int k, float* kernel, unsigned char* input, unsigned char* output){

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

	int row_in = row + head;
	float res = 1;	

	if (row <= M && col < N) {
		int MM = M + head;
		res = 0;
		int i = 0;
		int k2 = k/2;

                for (int f = (row_in-k2); f <= (row_in+k2); f++) {
                        for (int c = (col-(k2*4)); c <= (col+(k2*4)); c+=4) {
                                if (c >= 0 && c < N && f >= 0 && f <= MM) {
                      	                res += (kernel[i] * ((float) input[f*N + c]));
                                }
                                i = i + 1;
                        }
                }

                output[row*N+col] = (unsigned char) res;
	}
}



int main(int argc, char** argv) {
	//PNG inPng("blanc_10_10.png");
	PNG inPng("pixar.png");
	//PNG inPng("40_40_w.png");
	//PNG inPng("generate.png");
	PNG outPng;
	outPng.Create(inPng.w, inPng.h);

        const unsigned int w = inPng.w; 
        const unsigned int h = inPng.h;
        //const unsigned int N = (w > h) ? w : h;
        int size = w * h;
        int size4 = size*4;
	int k2 = KSIZE/2;

	printf("Heigh: %d \n Widht: %d \n\n", h, w);

	
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

	printf("k = %d\nrest = %d\n\n", k,rest);
	

	for (int j = 0; j < size4;) {
		dimensions[i][0] = j;
		dimensions[i][1] = (k + (i < rest))*inPng.w*4;
		dimensions[i][2] = dimensions[i][0] - k2*(i != 0)*inPng.w*4;
		dimensions[i][3] = dimensions[i][1] + (k2*(i != (NGPUS-1)) + k2*(i != 0))*inPng.w*4;

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
	float *d_kernel[NGPUS];
	char *h_output;

	cudaMallocHost((float**)&h_output,  size4*sizeof(unsigned char));

	float kernel[KSIZE*KSIZE] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};

        //Allocate memory in the devices
        for (int i = 0; i < NGPUS; i++){
                //Set the device
                cudaSetDevice(i);

                //Allocate memory in the device for the base image and the single component image
                cudaMalloc((void**)&d_input[i], dimensions[i][3] * sizeof(unsigned char));
		cudaMalloc((void**)&d_output[i], dimensions[i][1] * sizeof(unsigned char));
		cudaMalloc((void**)&d_kernel[i], KSIZE * KSIZE * sizeof(float));

                //Copy asynchronously the base image to the device
                cudaMemcpyAsync(d_input[i], &inPng.data[dimensions[i][2]], dimensions[i][3] * sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_kernel[i], &kernel[0], KSIZE * KSIZE * sizeof(float), cudaMemcpyHostToDevice);
        }


	int krest = k + (rest != 0);
	const unsigned int N = (w > krest) ? w*4 : krest*4;
        unsigned int nBlocks, nThreads;
        nThreads = SIZE;
        nBlocks = N/nThreads + (N%nThreads != 0);
        dim3 dimGrid(nBlocks, nBlocks, 1);
        dim3 dimBlock(nThreads, nThreads, 1);

	//float kernel[KSIZE*KSIZE] = {0,0,0,0,1,0,0,0,0};
	
	for (int i = 0; i < NGPUS; i++){
                //Set the device
                cudaSetDevice(i);
		convolution<<<dimGrid,dimBlock>>> ((i != 0)*k2, w*4, dimensions[i][1]/(w*4), KSIZE, d_kernel[i], (unsigned char*)d_input[i], (unsigned char*)d_output[i]);
        }

	
	//char* result[4];
		
	for (int i = 0; i < NGPUS; i++) {
		/*
		cudaMallocHost((float**)&result[i], dimensions[i][1]);
		cudaSetDevice(i);
                cudaMemcpy(result[i], d_output[i], dimensions[i][1], cudaMemcpyDeviceToHost);

                printf("Copying from the device %d from the position %d to the %d\n", i, dimensions[i][0], dimensions[i][0]+dimensions[i][1]);
                print_matrix((unsigned char*) result[i], dimensions[i][1], 4, inPng.w*4);
		*/
		
		cudaSetDevice(i);
		cudaMemcpy(&h_output[dimensions[i][0]], d_output[i], dimensions[i][1], cudaMemcpyDeviceToHost);
		printf("Copying from the device %d from the position %d to the %d\n", i, dimensions[i][0], dimensions[i][0]+dimensions[i][1]);
		print_matrix((unsigned char*) &h_output[dimensions[i][0]], dimensions[i][1], 4, inPng.w*4);
		
	}
	

		

	

        cudaDeviceSynchronize();
        
	std::copy(&h_output[0], &h_output[size4], std::back_inserter(outPng.data));
        outPng.Save("result.png");
	
	//print_matrix((unsigned char*) &h_output[dimensions[1][0]], size, 4, inPng.w*4);
	//print_matrix(&outPng.data[dimensions[0][1]], size, 4, dimensions[1][1]);
	

	/*		
	cudaMemcpyAsync(&h_output[dimensions[1][2]], d_input[1], dimensions[1][3], cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	print_matrix((unsigned char*) &h_output[dimensions[1][2]], dimensions[1][3], 4, inPng.w*4);
	*/

	return 0;

}
