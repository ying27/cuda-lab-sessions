#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>

#include "PNG.h"

#define NGPUS 4

#ifndef KSIZE
#define KSIZE 3
#endif


#ifndef SIZE
#define SIZE 32
#endif


void convolution (int row, int col, int N, int M, int k, float* kernel, unsigned char* input, unsigned char* output) {
	
	int i = 0;
	int k2 = k/2;
	int res;
	
        for (int f = (row-k2); f <= (row+k2); f++) {
        	for (int c = (col-(k2*4)); c <= (col+(k2*4)); c+=4) {
                	if (c >= 0 && c < N && f >= 0 && f < M) {
                      		res += (kernel[i] * ((float) input[f*N + c]));
                        }
                        i = i + 1;
                }
	}
        output[row*N+col] = (unsigned char) res;
}


int main(int argc, char** argv) {
	PNG inPng("widowmaker.png");
	//PNG inPng("pixar.png");
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
	unsigned char aux[size4];

	float kernel[9] = {0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.1f};
	std::clock_t    start;

     	start = std::clock();

	for (int row = 0; row < h; row++) {
		for (int col = 0; col < (w*4); col++) {
			convolution (row, col, w*4, h, KSIZE, kernel, &inPng.data[0], &aux[0]);
		}
	}

	std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

	std::copy(&aux[0], &aux[size4], std::back_inserter(outPng.data));
        outPng.Save("result.png");
	return 0;
}
