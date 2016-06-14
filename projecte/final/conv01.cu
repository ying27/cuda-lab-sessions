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

//N: numero de columnes en components
//M: numero de files

//N && M correspon al tamany de la matriu de output
__global__ void convolution (int head, int tail, int N, int M, int k, float* kernel, unsigned char* input, unsigned char* output){

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

	int row_in = row + head;
	float res = 1;	

	if (row <= M && col < N) {
		int MM = M + head + tail;
		res = 0;
		int i = 0;
		int k2 = k/2;

                for (int f = (row_in-k2); f <= (row_in+k2); f++) {
                        for (int c = (col-(k2*4)); c <= (col+(k2*4)); c+=4) {
                                if (c >= 0 && c < N && f >= 0 && f < MM) {
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
	//PNG inPng("widowmaker.png");
	PNG outPng;
	outPng.Create(inPng.w, inPng.h);

        const unsigned int w = inPng.w; 
        const unsigned int h = inPng.h;
        int size = w * h;
        int size4 = size*4;
	int k2 = KSIZE/2;

	//posicio [0]: Head / [1]: numero bytes / [2] = head + offset / [3] = bytes + offset
	int dimensions[NGPUS][4];
	int k = (inPng.h/NGPUS);
	int rest = inPng.h-(k*NGPUS);
	int i = 0;

	for (int j = 0; j < size4;) {
		dimensions[i][0] = j;
		dimensions[i][1] = (k + (i < rest))*inPng.w*4;
		dimensions[i][2] = dimensions[i][0] - k2*(i != 0)*inPng.w*4;
		dimensions[i][3] = dimensions[i][1] + (k2*(i != (NGPUS-1)) + k2*(i != 0))*inPng.w*4;

		j = j + (k + (i < rest))*inPng.w*4;
		i++;
	}
	
	char *d_input[NGPUS];
	char *d_output[NGPUS];
	float *d_kernel[NGPUS];
	char *h_output;

	cudaMallocHost((float**)&h_output,  size4*sizeof(unsigned char));

	float kernel[KSIZE*KSIZE] = {0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.1f};
	//float kernel[KSIZE*KSIZE] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f};


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
	
	for (int i = 0; i < NGPUS; i++){
                //Set the device
                cudaSetDevice(i);
		convolution<<<dimGrid,dimBlock>>> ((i != 0)*k2, (i != (NGPUS-1))*k2, w*4, dimensions[i][1]/(w*4), KSIZE, d_kernel[i], (unsigned char*)d_input[i], (unsigned char*)d_output[i]);
        }

	for (int i = 0; i < NGPUS; i++) {
		cudaSetDevice(i);
		cudaMemcpy(&h_output[dimensions[i][0]], d_output[i], dimensions[i][1], cudaMemcpyDeviceToHost);
	}

        cudaDeviceSynchronize();
        
	std::copy(&h_output[0], &h_output[size4], std::back_inserter(outPng.data));
        outPng.Save("result.png");
	
	return 0;

}
