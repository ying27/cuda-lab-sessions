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

__global__ void Kernel07(float *g_idata, float *g_odata, int N) {
  __shared__ float sdata[512];
  unsigned int s;

  // Cada thread realiza la suma parcial de los datos que le
  // corresponden y la deja en la memoria compartida
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  unsigned int gridSize = blockDim.x*2*gridDim.x;
  sdata[tid] = 0;
  while (i < N) {
    sdata[tid] += g_idata[i] + g_idata[i+blockDim.x];
    i += gridSize;
  }
  __syncthreads();

  // Hacemos la reduccion en la memoria compartida
  for (s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  // desenrrollamos el ultimo warp activo
  if (tid < 32) {
    volatile float *smem = sdata;

    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
  }


  // El thread 0 escribe el resultado de este bloque en la memoria global
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}


void InitV(int N, float *v);
int Test(int N, float *v, float sum);

int main(int argc, char** argv)

{
  unsigned int N;
  unsigned int numBytesV, numBytesW;
  unsigned int nBlocks, nThreads;
  int test;
  float elapsedTime;
  float t1,t2; 

  cudaEvent_t start, stop;

  float *h_v, *h_w;
  float *d_v, *d_w;
  float SUM;
  int i;

  N = 1024 * 1024 * 16;
  nThreads = 512;  // Este valor ha de coincidir con NumElementos

  // Numero maximo de Block Threads = 65535
  nBlocks = 4096;
  
  numBytesV = N * sizeof(float);
  numBytesW = nBlocks * sizeof(float);


  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Obtener Memoria en el host
  h_v = (float*) malloc(numBytesV); 
  h_w = (float*) malloc(numBytesW); 

  // Obtiene Memoria [pinned] en el host
  // cudaMallocHost((float**)&h_v, numBytesV); 
  // cudaMallocHost((float**)&h_w, numBytesW); 

  // Inicializa los vectores
  InitV(N, h_v);

  
  // Obtener Memoria en el device
  cudaMalloc((float**)&d_v, numBytesV); 
  cudaMalloc((float**)&d_w, numBytesW); 

  // Copiar datos desde el host en el device 
  cudaMemcpy(d_v, h_v, numBytesV, cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);

  // Ejecutar el kernel 
  Kernel07<<<nBlocks, nThreads>>>(d_v, d_w, N);

  // Obtener el resultado parcialdesde el host 
  cudaMemcpy(h_w, d_w, numBytesW, cudaMemcpyDeviceToHost); 


  SUM = 0.0;
  for (i=0; i<nBlocks; i++)
    SUM = SUM + h_w[i];

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Liberar Memoria del device 
  cudaFree(d_v);
  cudaFree(d_w);
 
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("\nKERNEL 07\n");
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

