#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "PNG.h"


#ifndef KSIZE
#define KSIZE 9
#endif


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






__global__ void convolution (const int N, const int M, const int K, float* kern, unsigned char* input, unsigned char* output){

        //extern __shared__ float sK[];
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
	float res = 0;
	int i = 0;

	//float kern[9] = {0,0,0,0,1,0,0,0,0};	
	
	if (row < M && col < N) {
		int k2 =K/2;
		for (int f = (row-k2); f <= (row+k2); f++) {
                	for (int c = (col-k2); c <= (col+k2); c++) {	
				if (c >= 0 && c < N && f >= 0 && f < M) {
					//res += (unsigned char) 20;
					res += kern[i] * ((float) input[f*N + c]);
				}
				i = i + 1;
			}
		}
		//output[row*N+col] = (unsigned char) res;
		output[row*N+col] = res;
	}
	
	//if (row < M && col < N) output[row*N+col] = input[row*N+col];

}







int main(int argc, char** argv)
{
	//PNG inPng("blanc_10_10.png");
	PNG inPng("pixar.png");
	//PNG inPng("40_40_w.png");
	PNG outPng;
	outPng.Create(inPng.w, inPng.h);

        const unsigned int w = inPng.w; 
        const unsigned int h = inPng.h;
        const unsigned int N = (w > h) ? w : h;
        int size = w * h * sizeof(unsigned char);  
        int size4 = size*4;

//	print_matrix(&inPng.data[0], size4, 4, inPng.w*4);


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
		cudaMemcpyAsync(d_base[i], &inPng.data[0], size4, cudaMemcpyHostToDevice);
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

        char* d_output[4];
        //float kernel[9] = {1/10, 1/10, 1/10, 1/10, 2/10, 1/10, 1/10, 1/10, 1/10};
	//float kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
	//float kernel[9] = {1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9};
	float kernel[9] = {0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.1f};
	//float kernel[9] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float* d_kernel[4];

        for (int i = 0; i < 4; i++){
                //Set the device
                cudaSetDevice(i);
                cudaMalloc((float**)&d_output[i], size);
                cudaMalloc((float**)&d_kernel[i], 9*sizeof(float));
                cudaMemcpy(d_kernel[i], &kernel[0], 9*sizeof(float), cudaMemcpyHostToDevice);
                
                convolution<<<dimGrid,dimBlock>>> (w, h, 3, d_kernel[i], (unsigned char*)d_components[i], (unsigned char*)d_output[i]);
                
                cudaFree(d_components[i]);
                cudaFree(d_kernel[i]);
        }




	/***********************************************************/

        //Get the processed component
        for (int i = 0; i < 3; i++) {
                cudaSetDevice(i);
                cudaMemcpyAsync(h_components[i], d_output[i], size, cudaMemcpyDeviceToHost);
                cudaFree(d_components[i]);

                cudaSetDevice(3);
                cudaMalloc((void**)&d_components[i], size);
                cudaMemcpyAsync(d_components[i], h_components[i], size, cudaMemcpyHostToDevice);
        }


	cudaMalloc((void**)&d_base[0], size4);

	join_components<<<dimGrid,dimBlock>>> (w, h, (unsigned char*) d_components[0], (unsigned char*) d_components[1], 
						     (unsigned char*) d_components[2], (unsigned char*) d_output[3], 
						     (unsigned char*) d_base[0]);


	cudaSetDevice(3);
        cudaError_t cudaStatus;
        cudaMallocHost((float**)&temp,  size4);
        cudaStatus = cudaMemcpyAsync(temp, d_base[0], size4, cudaMemcpyDeviceToHost);

        if (cudaStatus != cudaSuccess)
        {
                std::cout << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                exit(1);
        }

        cudaStatus = cudaDeviceSynchronize();
        std::copy(&temp[0], &temp[w*h*4], std::back_inserter(outPng.data));
        outPng.Save("result.png");
	print_matrix(&outPng.data[0], size4, 4, inPng.w*4);


/*
	cudaFreeHost(h_components[0]);
	for (int i = 0; i < 4; i++){
		cudaSetDevice(i);
		cudaFree(d_components[i]);
	}
*/
	return 0;

}
