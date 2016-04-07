#include <stdio.h>
#include <stdlib.h>


#include <sys/times.h>
#include <sys/resource.h>

float GetTime(void)        
{
  struct timeval tim;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  tim=ru.ru_utime;
  return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}

__global__ void Kernel04(float *g_idata, float *g_odata) { 
  __shared__ float sdata[512];
  unsigned int s;

  // Cada thread carga 1 elemento desde la memoria global
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
  sdata[tid] = g_idata[i];
  __syncthreads();

  // Hacemos la reduccion en la memoria compartida
  for (s=blockDim.x/2; s>0; s>>=1) { 
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads(); 
  }


  // El thread 0 escribe el resultado de este bloque en la memoria global
  if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 

}


void InitV(int N, float *v);
int Test(int N, float *v, float sum);

int main(int argc, char** argv)

{
  unsigned int N;
  unsigned int numBytesV, numBytesW, numBytesX;
  unsigned int nBlocks, nThreads;
  int test;
  float elapsedTime;
  float t1,t2; 

  cudaEvent_t start, stop;

  float *h_v, *h_w, *h_x;
  float *d_v, *d_w, *d_x;
  float SUM;
  int i;

  N = 1024 * 1024 * 16;
  nThreads = 512;  // Este valor ha de coincidir con NumElementos

  // Numero maximo de Block Threads = 65535
  nBlocks = N/nThreads;  // Solo funciona bien si N multiplo de nThreads
  
  numBytesV = N * sizeof(float);
  numBytesW = nBlocks * sizeof(float);
  numBytesX = (nBlocks/nThreads) * sizeof(float);


  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Obtener Memoria en el host
  h_v = (float*) malloc(numBytesV); 
  h_w = (float*) malloc(numBytesW); 
  h_x = (float*) malloc(numBytesX);

  // Obtiene Memoria [pinned] en el host
  // cudaMallocHost((float**)&h_v, numBytesV); 
  // cudaMallocHost((float**)&h_w, numBytesW); 
  // cudaMallocHost((float**)&h_x, numBytesX); 

  // Inicializa los vectores
  InitV(N, h_v);

  
  // Obtener Memoria en el device
  cudaMalloc((float**)&d_v, numBytesV); 
  cudaMalloc((float**)&d_w, numBytesW); 
  cudaMalloc((float**)&d_x, numBytesX);

  // Copiar datos desde el host en el device 
  cudaMemcpy(d_v, h_v, numBytesV, cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);

  // Ejecutar el kernel 
  Kernel04<<<nBlocks, nThreads>>>(d_v, d_w);
  Kernel04<<<nBlocks/nThreads, nThreads>>>(d_w, d_x);

  // Obtener el resultado parcial desde el host 
  cudaMemcpy(h_x, d_x, numBytesX, cudaMemcpyDeviceToHost);


  SUM = 0.0;
  for (i=0; i<(nBlocks/nThreads); i++)
    SUM = SUM + h_x[i];

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Liberar Memoria del device 
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_x);
 
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("\nKERNEL 04\n");
  printf("Vector Size: %d\n", N);
  printf("nThreads: %d\n", nThreads);
  printf("nBlocks: %d\n", nBlocks);
  printf("Tiempo Total %4.6f ms\n", elapsedTime);
  printf("Ancho de Banda %4.3f GB/s\n", (N * sizeof(float)) / (1000000 * elapsedTime));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  t1=GetTime();
  test = Test(N, h_v, SUM);
  t2=GetTime();

  if (test)
    printf ("TEST PASS, Time seq: %f ms\n", t2-t1);
  else
    printf ("TEST FAIL\n");

}


void InitV(int N, float *v) {
   int i;
   for (i=0; i<N; i++) 
     v[i] = rand() / (float) RAND_MAX;
   
}

int error(float a, float b) {
  float tmp;

  tmp = abs(a-b) / abs(min(a,b));

  if (tmp > 0.0001) return 1;
  else  return 0;

}

int Test(int N, float *v, float sum) {
   int i;
   float tmp;

   tmp = 0.0;
   for (i=0; i<N; i++) 
     tmp = tmp + v[i];
   if (error(tmp, sum)) {
     printf ("ERROR: %f - %f = %f \n", tmp, sum, abs(tmp - sum));
     return 0;
   }
   return 1;
}

