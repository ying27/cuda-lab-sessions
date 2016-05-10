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
		cudaMemcpyAsync(d_base[i], &inPng.data[0], size4, cudaMemcpyHostToDevice);
	}
	

	cudaSetDevice(0);
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

/*
	//Extract the components of base image
	for (int i = 0; i < 4; i++){
		//Set the device
		cudaSetDevice(i);
		extract_component<<<dimGrid,dimBlock>>> (i, w, h, (unsigned char*) d_base[i], (unsigned char*) d_components[i]);
		//cudaFree(d_base[i]);
	}

	//Get the processed component
	for (int i = 1; i < 4; i++) {
	        cudaSetDevice(i);
        	cudaMemcpyAsync(h_components[i], d_components[i], size, cudaMemcpyDeviceToHost);
        	cudaFree(d_components[i]);

        	cudaSetDevice(0);
        	cudaMalloc((void**)&d_components[i], size);
        	cudaMemcpyAsync(d_components[i], h_components[i], size, cudaMemcpyHostToDevice);	
	}

	cudaSetDevice(0);
*/
	














/*
	print_matrix((unsigned char*)h_components[0], size, 1, inPng.w);	
        print_matrix((unsigned char*)h_components[1], size, 1, inPng.w);
        print_matrix((unsigned char*)h_components[2], size, 1, inPng.w);
        print_matrix((unsigned char*)h_components[3], size, 1, inPng.w);
*/


/*
	cudaMalloc((void**)&d_base[0], size4);

	join_components<<<dimGrid,dimBlock>>> (w, h, (unsigned char*) d_components[0], (unsigned char*) d_components[1], 
						     (unsigned char*) d_components[2], (unsigned char*) d_components[3], 
						     (unsigned char*) d_base[0]);
*/


	//auto tmp = new unsigned char[w * h * 4];
	// Copy output vector from GPU buffer to host memory.
	//cudaMemcpy(tmp, d_base[0], size4, cudaMemcpyDeviceToHost);
	//std::copy(&tmp[0], &tmp[w * h * 4], std::back_inserter(outPng.data));
	


	//cudaMallocHost((float**)h_modified,  size4);
/*
	cudaError_t cudaStatus;
	auto h_modified = new unsigned char[w*h*4];
	cudaStatus = cudaMemcpy(h_modified, d_base[0], w*h*4, cudaMemcpyDeviceToHost);

    	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		exit(1);
    	}
*/


	//cudaMemcpyAsync(&outPng.data[0], d_base[0], size4, cudaMemcpyDeviceToHost);

/*	
        std::copy(&h_modified[0], &h_modified[size4], std::back_inserter(outPng.data));

	//outPng.Save("cuda_tutorial_2.png");	
	
	print_matrix(&outPng.data[0], size4, 4, inPng.w*4);
	//outPng.Save("result.png");

*/

/*
	cudaFreeHost(h_components[0]);
	for (int i = 0; i < 4; i++){
		cudaSetDevice(i);
		cudaFree(d_components[i]);
	}
*/
	return 0;

}
