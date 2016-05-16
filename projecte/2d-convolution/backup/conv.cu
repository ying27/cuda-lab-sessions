#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "PNG.h"

#ifndef SIZE
#define SIZE 32
#endif


//N: numero de columnes en pixels (cada pixel son 4 components)
//M: numero de files en pixels (cada pixel son 4 components)

__global__ void join_components (unsigned int N, unsigned int M,  
				 unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* a,
				 unsigned char* joined){

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
	int N4 = N*4;
	int col4 = col*4;

        if (row < M && col < N) {
                joined[row*N4+col4] = r[row*N+col];
		joined[row*N4+col4+1] = g[row*N+col];
		joined[row*N4+col4+2] = b[row*N+col];
		joined[row*N4+col4+3] = a[row*N+col];
	}
}


//N: numero de columnes
//M: numero de files
__global__ void extract_component (int comp, unsigned int N, unsigned int M, unsigned char* color_mat, unsigned char* single_mat){

	int row = blockIdx.y * blockDim.y + threadIdx.y;
  	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int col4 = col*4+comp;
	int N4 = N*4;

	if (row < M && col < N) 
		single_mat[row*N+col] = color_mat[row*N4+col4];
}



//Kernel<<< gridDim, blockDim, SharedMemorySize >>>(count)
__global__ void convolution (const int N, const int M, const int K, char* kernel, unsigned char* input, unsigned char* output){

	//extern __shared__ float sK[];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
	char res = 0;
	/*
	if (blockIdx.y == 0 && blockIdx.x == 0) {
		for (int i = 0 ; i < K*K; i++) {
			sK[i] = kernel[i];
		}
	}
	__syncthreads();
	*/
/*
	//float *p = &sK[0];
	char *p = &kernel[0];
	int k2 = K/2;
	for (int f = (row-k2); f < (row+k2); f++) {
		for (int c = (col-k2) ; c < (col+k2); c++) {
			if (f > 0 && f < M && c > 0 && c < N) res += input[f*N+c] * (*p);			
			p++;
		}
	}
*/
	//output[row*N+col] = res;
	if (row < M && col < N) output[row*N+col] = input[row*N+col];
}



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


int main(int argc, char** argv)
{
	//PNG inPng("blanc_10_10.png");
	PNG inPng("pixar.png");
	PNG outPng;
	outPng.Create(inPng.w, inPng.h);

        const unsigned int w = inPng.w; 
        const unsigned int h = inPng.h;
        const unsigned int N = (w > h) ? w : h;
        int size = w * h * sizeof(unsigned char);  
        int size4 = size*4;

	print_matrix(&inPng.data[0], size4, 4, inPng.w*4);


	unsigned int nBlocks, nThreads;
	nThreads = SIZE;
	nBlocks = N/nThreads + (N%nThreads != 0);
	dim3 dimGrid(nBlocks, nBlocks, 1);
	dim3 dimBlock(nThreads, nThreads, 1);


	char *d_base[4];
	char *d_components[4];
	char *h_components[4];
	char *temp;

	cudaMallocHost((float**)&h_components[0],  size);
	cudaMallocHost((float**)&h_components[1],  size);
	cudaMallocHost((float**)&h_components[2],  size);
	cudaMallocHost((float**)&h_components[3],  size);

	//Allocate memory in the devices
	for (int i = 0; i < 4; i++){
		//Set the device
		cudaSetDevice(i);
		//Allocate memory in the device for the base image and the single component image
        	cudaMalloc((void**)&d_base[i], size4);
		cudaMalloc((void**)&d_components[i], size);
		//Copy asynchronously the base image to the device
		cudaMemcpy(d_base[i], &inPng.data[0], size4, cudaMemcpyHostToDevice);
	}
	
	//Extract the components of base image
	for (int i = 0; i < 4; i++){
		//Set the device
		cudaSetDevice(i);
		extract_component<<<dimGrid,dimBlock>>> (i, w, h, (unsigned char*) d_base[i], (unsigned char*) d_components[i]);
		cudaFree(d_base[i]);
	}

	/************************************************************/
 	//Insert Work	
	/*
	char* d_output[4];
	char kernel[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
	char* d_kernel[4];

	for (int i = 0; i < 4; i++){
                //Set the device
                cudaSetDevice(i);
		cudaMallocHost((float**)&d_output[i], size);
		cudaMallocHost((float**)&d_kernel[i], 9);
		cudaMemcpy(d_kernel[i], &kernel, 9*sizeof(char), cudaMemcpyHostToDevice);
		
		convolution<<<dimGrid,dimBlock>>> (w, h, 3, d_kernel[i], (unsigned char*)d_components[i], (unsigned char*)d_output[i]);
		
		cudaFree(d_components[i]);
		cudaFree(d_kernel[i]);
        }
	*/
	/***********************************************************/

        //Get the processed component
        for (int i = 1; i < 4; i++) {
                cudaSetDevice(i);
                //cudaMemcpy(h_components[i], d_output[i], size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_components[i], d_components[i], size, cudaMemcpyDeviceToHost);
                cudaFree(d_output[i]);

                cudaSetDevice(0);
                cudaMalloc((void**)&d_components[i], size);
                cudaMemcpy(d_components[i], h_components[i], size, cudaMemcpyHostToDevice);
        }


	cudaMalloc((void**)&d_base[0], size4);

	join_components<<<dimGrid,dimBlock>>> (w, h, (unsigned char*) d_components[0], (unsigned char*) d_components[1], 
						     (unsigned char*) d_components[2], (unsigned char*) d_components[3], 
						     (unsigned char*) d_base[0]);


	cudaSetDevice(0);
        cudaError_t cudaStatus;
        cudaMallocHost((float**)&temp,  size4);
        cudaStatus = cudaMemcpy(temp, d_base[0], size4, cudaMemcpyDeviceToHost);

        if (cudaStatus != cudaSuccess)
        {
                std::cout << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                exit(1);
        }

        cudaStatus = cudaDeviceSynchronize();
        std::copy(&temp[0], &temp[w*h*4], std::back_inserter(outPng.data));
        outPng.Save("result.png");

	print_matrix(&outPng.data[0], size4, 4, outPng.w*4);

	cudaFreeHost(h_components[0]);
	for (int i = 0; i < 4; i++){
		cudaSetDevice(i);
		cudaFree(d_components[i]);
	}

	return 0;

}
