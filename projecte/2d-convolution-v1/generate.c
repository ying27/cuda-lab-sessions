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
			if (i%w == 0) printf("\n");
		}
	}
	printf("\n\n");
}
 
int main(int argc, char** argv)
{
	PNG outPng;
	outPng.Create(30, 30);

	unsigned char matrix[120][120];
	unsigned char cont = 0;


	for (int j = 0;j < 120; j++){
		for (int i = 0; i < 120; i++){
			matrix[j][i] = cont;
			cont++;
			cont = cont%240;
		}
	}

	std::copy(&matrix[0][0], &matrix[119][119], std::back_inserter(outPng.data));
        outPng.Save("generate.png");

	print_matrix(&matrix[0][0], 120*120, 4, 120);

	return 0;
}
