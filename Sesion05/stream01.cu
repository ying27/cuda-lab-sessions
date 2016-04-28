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

  cudaStream_t stream1, stream2, stream3, stream4;

  char test;

  // Dimension del vector y comprobacion resultado
  if (argc == 1)      { test = 'N';      N = 1024 * 2048; }
  else if (argc == 2) { test = 'N';      N = 1024 * atoi(argv[1]); }
  else if (argc == 3) { test = *argv[2]; N = 1024 * atoi(argv[1]); }
  else { printf("Usage: ./exe TAM test\n"); exit(0); }

  // numero de Threads
  nThreads = 1024;

  numBytes = N * sizeof(float);
  int numBytes4 = numBytes/4;
  int N4 = N/4;

  // numero de Blocks en cada dimension 
  nBlocks = (N4+nThreads-1)/nThreads; 

  cudaEventCreate(&E0);
  cudaEventCreate(&E3);

  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  cudaStreamCreate(&stream4);

  // Obtiene Memoria [pinned] en el host
  cudaMallocHost((float**)&h_A, numBytes); 
  cudaMallocHost((float**)&h_B, numBytes); 
  cudaMallocHost((float**)&h_C, numBytes); 

  // Inicializa las matrices
  InitV(N, h_A);
  InitV(N, h_B);

  // Obtener Memoria en el device
  cudaMalloc((float**)&d_A, numBytes); 
  cudaMalloc((float**)&d_B, numBytes); 
  cudaMalloc((float**)&d_C, numBytes); 

  cudaEventRecord(E0, 0);
  
  for (times=0; times<TIMES; times++) {

    cudaMemcpyAsync(&d_A[0], &h_A[0], numBytes4, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(&d_B[0], &h_B[0], numBytes4, cudaMemcpyHostToDevice, stream1);
    Kernel00<<<nBlocks, nThreads, 0, stream1>>>(N4, a, b, &d_A[0], &d_B[0], &d_C[0]);
    cudaMemcpyAsync(&h_C[0], &d_C[0], numBytes4, cudaMemcpyDeviceToHost, stream1); 

    cudaMemcpyAsync(&d_A[N4], &h_A[N4], numBytes4, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(&d_B[N4], &h_B[N4], numBytes4, cudaMemcpyHostToDevice, stream2);
    Kernel00<<<nBlocks, nThreads, 0, stream2>>>(N4, a, b, &d_A[N4], &d_B[N4], &d_C[N4]);
    cudaMemcpyAsync(&h_C[N4], &d_C[N4], numBytes4, cudaMemcpyDeviceToHost, stream2); 

    cudaMemcpyAsync(&d_A[2*N4], &h_A[2*N4], numBytes4, cudaMemcpyHostToDevice, stream3);
    cudaMemcpyAsync(&d_B[2*N4], &h_B[2*N4], numBytes4, cudaMemcpyHostToDevice, stream3);
    Kernel00<<<nBlocks, nThreads, 0, stream3>>>(N4, a, b, &d_A[2*N4], &d_B[2*N4], &d_C[2*N4]);
    cudaMemcpyAsync(&h_C[2*N4], &d_C[2*N4], numBytes4, cudaMemcpyDeviceToHost, stream3); 

    cudaMemcpyAsync(&d_A[3*N4], &h_A[3*N4], numBytes4, cudaMemcpyHostToDevice, stream4);
    cudaMemcpyAsync(&d_B[3*N4], &h_B[3*N4], numBytes4, cudaMemcpyHostToDevice, stream4);
    Kernel00<<<nBlocks, nThreads, 0, stream4>>>(N4, a, b, &d_A[3*N4], &d_B[3*N4], &d_C[3*N4]);
    cudaMemcpyAsync(&h_C[3*N4], &d_C[3*N4], numBytes4, cudaMemcpyDeviceToHost, stream4); 

/*
    cudaMemcpyAsync(&d_A[0], &h_A[0], numBytes4, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(&d_A[N4], &h_A[N4], numBytes4, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(&d_A[2*N4], &h_A[2*N4], numBytes4, cudaMemcpyHostToDevice, stream3);
    cudaMemcpyAsync(&d_A[3*N4], &h_A[3*N4], numBytes4, cudaMemcpyHostToDevice, stream4);

    cudaMemcpyAsync(&d_B[0], &h_B[0], numBytes4, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(&d_B[N4], &h_B[N4], numBytes4, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(&d_B[2*N4], &h_B[2*N4], numBytes4, cudaMemcpyHostToDevice, stream3);
    cudaMemcpyAsync(&d_B[3*N4], &h_B[3*N4], numBytes4, cudaMemcpyHostToDevice, stream4);

    Kernel00<<<nBlocks, nThreads, 0, stream1>>>(N4, a, b, &d_A[0], &d_B[0], &d_C[0]);
    Kernel00<<<nBlocks, nThreads, 0, stream2>>>(N4, a, b, &d_A[N4], &d_B[N4], &d_C[N4]);
    Kernel00<<<nBlocks, nThreads, 0, stream3>>>(N4, a, b, &d_A[2*N4], &d_B[2*N4], &d_C[2*N4]);
    Kernel00<<<nBlocks, nThreads, 0, stream4>>>(N4, a, b, &d_A[3*N4], &d_B[3*N4], &d_C[3*N4]);

    cudaMemcpyAsync(&h_C[0], &d_C[0], numBytes4, cudaMemcpyDeviceToHost, stream1); 
    cudaMemcpyAsync(&h_C[N4], &d_C[N4], numBytes4, cudaMemcpyDeviceToHost, stream2); 
    cudaMemcpyAsync(&h_C[2*N4], &d_C[2*N4], numBytes4, cudaMemcpyDeviceToHost, stream3); 
    cudaMemcpyAsync(&h_C[3*N4], &d_C[3*N4], numBytes4, cudaMemcpyDeviceToHost, stream4); 
*/
  }

  cudaEventRecord(E3, 0); cudaEventSynchronize(E3);

  // Liberar Memoria del device 
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaEventElapsedTime(&TiempoTotal,  E0, E3);
  printf("\nKERNEL 01: 4 Streams\n");
  printf("Dimension Problema: %d\n", N);
  printf("Invocacion Kernel <<<nBlocks, nKernels>>> (N): <<<%d, %d>>> (%d)\n", nBlocks, nThreads, N4);
  printf("nKernels: %d\n", 4);

  if (PINNED) printf("Usando Pinned Memory\n");
         else printf("NO usa Pinned Memory\n");
  printf("Tiempo Global (01): %4.6f milseg\n", TiempoTotal);
  printf("Rendimiento Global (01): %4.2f GFLOPS\n", (TIMES*(3.0 + 4.0*DUMMY) * (float) N) / (1000000.0 * TiempoTotal));

  cudaEventDestroy(E0); cudaEventDestroy(E3);
  cudaStreamDestroy(stream1); cudaStreamDestroy(stream2); cudaStreamDestroy(stream3); cudaStreamDestroy(stream4);


  if (test == 'N')
    printf ("NO TEST\n");
  else  if (TestsumV(N, a, b, h_A, h_B, h_C))
    printf ("TEST PASS\n");
  else
    printf ("TEST FAIL\n");

  cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);

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

