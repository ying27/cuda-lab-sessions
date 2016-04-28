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

#define nStreams 4

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

  cudaStream_t stream[nStreams];
  int str;

  char test;

  // Dimension del vector y comprobacion resultado
  if (argc == 1)      { test = 'N';      N = 1024 * 2048; }
  else if (argc == 2) { test = 'N';      N = 1024 * atoi(argv[1]); }
  else if (argc == 3) { test = *argv[2]; N = 1024 * atoi(argv[1]); }
  else { printf("Usage: ./exe TAM test\n"); exit(0); }

  numBytes = N * sizeof(float);

  cudaEventCreate(&E0);
  cudaEventCreate(&E3);

  for (str=0; str<nStreams; str++) 
    cudaStreamCreate(&stream[str]);

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
  
  int punt, nB, nK;
  nK = 2500*1024;
  nB = nK * sizeof(float);

  // numero de Threads
  nThreads = 1024;

  // numero de Blocks en cada dimension 
  nBlocks = nK / nThreads; 

  for (times=0; times<TIMES; times++) {

    for (str=0, punt=0; punt < N; str = (str+1)%nStreams, punt += nK) {
      cudaMemcpyAsync(&d_A[punt], &h_A[punt], nB, cudaMemcpyHostToDevice, stream[str]);
      cudaMemcpyAsync(&d_B[punt], &h_B[punt], nB, cudaMemcpyHostToDevice, stream[str]);
      Kernel00<<<nBlocks, nThreads, 0, stream[str]>>>(nK, a, b, &d_A[punt], &d_B[punt], &d_C[punt]);
      cudaMemcpyAsync(&h_C[punt], &d_C[punt], nB, cudaMemcpyDeviceToHost, stream[str]); 
    }

  }

  cudaEventRecord(E3, 0); cudaEventSynchronize(E3);

  // Liberar Memoria del device 
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaEventElapsedTime(&TiempoTotal,  E0, E3);
  printf("\nKERNEL 02: Streams 2\n");
  printf("Dimension Problema: %d\n", N);
  printf("Invocacion Kernel <<<nBlocks, nKernels>>> (N): <<<%d, %d>>> (%d)\n", nBlocks, nThreads, nK);
  printf("nKernels: %d\n", N/nK);
  if (PINNED) printf("Usando Pinned Memory\n");
         else printf("NO usa Pinned Memory\n");
  printf("Tiempo Global (02): %4.6f milseg\n", TiempoTotal);
  printf("Rendimiento Global (02): %4.2f GFLOPS\n", (TIMES*(3.0 + 4.0*DUMMY) * (float) N) / (1000000.0 * TiempoTotal));

  cudaEventDestroy(E0); cudaEventDestroy(E3);

  for (str=0; str<nStreams; str++) 
    cudaStreamDestroy(stream[str]);


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

