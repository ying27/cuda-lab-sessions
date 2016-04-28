#include <stdio.h>
#include <stdlib.h>

#ifndef PINNED
#define PINNED 1
#endif

#ifndef DUMMY
#define DUMMY 30 
#endif

#ifndef TIMES
#define TIMES 5 
#endif

// Suma de Vectores ponderados
// C(N) <- a*A(N) + b*B(N)

__global__ void Kernel00 (int N, float a, float b, float *A, float *B, float *C) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;  
  if (i<N) {
      C[i] = a*A[i] + b*B[i];
      for (j=0; j<DUMMY; j++) 
        C[i] += a*A[i] + b*B[i];
  }
}



void InitV(int N, float *V);
int TestsumV(int N, float a, float b, float *A, float *B, float *C);


// Invocacion:
// ./ejecutable TAM test
// TAM es el la dimension del vector medido en K ( N = TAM * 1024)
// test == 'Y', comprueba que el resultado sea correcto
// test == 'N', NO comprueba que el resultado (Util para tomar tiempos)
// Por defecto, tam = 2048, test == 'N'

int main(int argc, char** argv)
{
  unsigned int N;
  unsigned int numBytes;
  unsigned int nBlocks, nThreads;
  unsigned int times;
 
  float TiempoTotal;
  cudaEvent_t E0, E3;

  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;
  float a = 0.3;
  float b = 0.7;

  char test;

  // Dimension del vector y comprobacion resultado
  if (argc == 1)      { test = 'N';      N = 1024 * 2048; }
  else if (argc == 2) { test = 'N';      N = 1024 * atoi(argv[1]); }
  else if (argc == 3) { test = *argv[2]; N = 1024 * atoi(argv[1]); }
  else { printf("Usage: ./exe TAM test\n"); exit(0); }

  // numero de Threads
  nThreads = 1024;

  // numero de Blocks en cada dimension 
  nBlocks = (N+nThreads-1)/nThreads; 
  
  numBytes = N * sizeof(float);


  cudaEventCreate(&E0);
  cudaEventCreate(&E3);

  if (PINNED) {
    // Obtiene Memoria [pinned] en el host
    cudaMallocHost((float**)&h_A, numBytes); 
    cudaMallocHost((float**)&h_B, numBytes); 
    cudaMallocHost((float**)&h_C, numBytes); 
  }
  else {
    // Obtener Memoria en el host
    h_A = (float*) malloc(numBytes); 
    h_B = (float*) malloc(numBytes); 
    h_C = (float*) malloc(numBytes); 
  }

  // Inicializa las matrices
  InitV(N, h_A);
  InitV(N, h_B);

  // Obtener Memoria en el device
  cudaMalloc((float**)&d_A, numBytes); 
  cudaMalloc((float**)&d_B, numBytes); 
  cudaMalloc((float**)&d_C, numBytes); 

  cudaEventRecord(E0, 0);
  
  for (times=0; times<TIMES; times++) {
    // Copiar datos desde el host en el device 
    cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numBytes, cudaMemcpyHostToDevice);
  
    // Ejecutar el kernel 
    Kernel00<<<nBlocks, nThreads>>>(N, a, b, d_A, d_B, d_C);

    // Obtener el resultado desde el host 
    cudaMemcpy(h_C, d_C, numBytes, cudaMemcpyDeviceToHost); 
  }

  cudaEventRecord(E3, 0); cudaEventSynchronize(E3);

  // Liberar Memoria del device 
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaEventElapsedTime(&TiempoTotal,  E0, E3);
  printf("\nKERNEL 00: NO - Streams\n");
  printf("Dimension Problema: %d\n", N);
  printf("Invocacion Kernel <<<nBlocks, nKernels>>> (N): <<<%d, %d>>> (%d)\n", nBlocks, nThreads, N);
  printf("nKernels: %d\n", 1);
  if (PINNED) printf("Usando Pinned Memory\n");
         else printf("NO usa Pinned Memory\n");
  printf("Tiempo Global (00): %4.6f milseg\n", TiempoTotal);
  printf("Rendimiento Global (00): %4.2f GFLOPS\n", (TIMES*(3.0 + 4.0*DUMMY) * (float) N) / (1000000.0 * TiempoTotal));

  cudaEventDestroy(E0); cudaEventDestroy(E3);


  if (test == 'N')
    printf ("NO TEST\n");
  else  if (TestsumV(N, a, b, h_A, h_B, h_C))
    printf ("TEST PASS\n");
  else
    printf ("TEST FAIL\n");

  if (PINNED) {
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
  }
  else {
    free(h_A); free(h_B); free(h_C);
  }

  cudaDeviceReset();

}


void InitV(int N, float *V) {
   int i;
   for (i=0; i<N; i++) 
     V[i] = rand() / (float) RAND_MAX;
   
}

int error(float a, float b) {
  float tmp;

  tmp = abs(a-b) / abs(min(a,b));

  if (tmp > 0.0001) return 1;
  else  return 0;

}

int TestsumV(int N, float a, float b, float *A, float *B, float *C) {
   int i, j;
   float tmp;
   for (i=0; i<N; i++) {
       tmp = a*A[i] + b*B[i]; 
      for (j=0; j<DUMMY; j++) 
        tmp += a*A[i] + b*B[i];
       if (error(tmp, C[i])) {
         printf ("%d: %f - %f = %f \n", i, tmp, C[i], abs(tmp - C[i]));
         return 0;
       }
   }
   
   return 1;
}

