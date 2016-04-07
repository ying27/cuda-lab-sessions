#include <stdio.h>
#include <stdlib.h>

__global__ void saxpyP (int N, float a, float *x, float *y) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  y[i] = a * x[i] + y[i];

}

void InitV(int N, float *v);
int TestSaxpy(int N, float a, float *x, float *y, float *Y);
void CheckCudaError(char sms[], int line);



int main(int argc, char** argv)

{
  unsigned int N;
  unsigned int numBytes;
  unsigned int nBlocks, nThreads;
 
  cudaEvent_t E0, E1, E2, E3, E4, E5;
  float TiempoTotal, TiempoKernel, TiempoMalloc;

  float *h_x, *h_y, *H_y;
  float *d_x, *d_y;

  N = 1024 * 1024 * 16;
  nThreads = 1024;
  nBlocks = N/nThreads;  // Solo funciona bien si N multiplo de nThreads
  numBytes = N * sizeof(float);


  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);
  cudaEventCreate(&E4);
  cudaEventCreate(&E5);

  // Obtener Memoria en el host
  h_x = (float*) malloc(numBytes); 
  h_y = (float*) malloc(numBytes); 
  H_y = (float*) malloc(numBytes);    // Solo se usa para comprobar el resultado

  // Obtiene Memoria [pinned] en el host
  //cudaMallocHost((float**)&h_x, numBytes); 
  //cudaMallocHost((float**)&h_y, numBytes); 
  //cudaMallocHost((float**)&H_y, numBytes);   // Solo se usa para comprobar el resultado

  // Inicializa los vectores
  InitV(N, h_x);
  InitV(N, h_y);

 
  cudaEventRecord(E0, 0);
  cudaEventSynchronize(E0);
 

  cudaEventRecord(E4, 0);
  cudaEventSynchronize(E4);

  // Obtener Memoria en el device
  cudaMalloc((float**)&d_x, numBytes); 
  cudaMalloc((float**)&d_y, numBytes);

  cudaEventRecord(E5, 0);
  cudaEventSynchronize(E5);

  CheckCudaError((char *) "Obtener Memoria en el device", __LINE__); 

  // Copiar datos desde el host en el device 
  cudaMemcpy(d_x, h_x, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, numBytes, cudaMemcpyHostToDevice);
  CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);

  // Ejecutar el kernel 
  saxpyP<<<nBlocks, nThreads>>>(N, 3.5, d_x, d_y);
  CheckCudaError((char *) "Invocar Kernel", __LINE__);

  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);

  // Obtener el resultado desde el host 
  // Guardamos el resultado en H_y para poder comprobar el resultado
  cudaMemcpy(H_y, d_y, numBytes, cudaMemcpyDeviceToHost); 
  CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  // Liberar Memoria del dev<F7><F8>ice 
  cudaFree(d_x); cudaFree(d_y);

  cudaDeviceSynchronize();

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);


  cudaEventElapsedTime(&TiempoTotal,  E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);
  cudaEventElapsedTime(&TiempoMalloc, E4, E5);
 

  printf("nThreads: %d\n", nThreads);
  printf("nBlocks: %d\n", nBlocks);

  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);

  printf("Tiempo cudaMalloc: %4.6f milseg\n", TiempoMalloc);

  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

  if (TestSaxpy(N, 3.5, h_x, h_y, H_y))
    printf ("TEST PASS\n");
  else
    printf ("TEST FAIL\n");

}


void InitV(int N, float *v) {
   int i;
   for (i=0; i<N; i++) 
     v[i] = rand();
   
}
int error(float a, float b) {

  if (abs (a - b) / a > 0.000001) return 1;
  else  return 0;

}

int TestSaxpy(int N, float a, float *x, float *y, float *Y) {
   int i;
   float tmp;

   for (i=0; i<N; i = i+17423) {
     tmp = a * x[i] + y[i];
     if (error(tmp, Y[i])) {
       printf ("%d: %f - %f = %f \n", i, tmp, Y[i], tmp - Y[i]);
       return 0;
     }
   }
   return 1;
}

void CheckCudaError(char sms[], int line) {
  cudaError_t error;
 
  error = cudaGetLastError();
  if (error) {
    printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
    exit(EXIT_FAILURE);
  }


}


