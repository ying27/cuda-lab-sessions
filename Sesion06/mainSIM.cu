#include <stdio.h>
#include <stdlib.h>

#define N 128
#define M 32
#define K 200

__device__ void entry(volatile int *foo) {
  for (int i=0; i<N; i++)
    atomicAdd((int *)foo, 1);
}

extern "C"
__global__ void diverge_cta(volatile int *foo) {
  __shared__ int x;
  if ((threadIdx.x % 32) !=0) {
    return;
  }
  entry(foo);

  if (threadIdx.x == 0) {
    x = 5;
    return;
  }
  __syncthreads();
  atomicAdd((int *)foo, x);
}



int main(int argc, char** argv)
{
  int *foo;
  int h_foo;

  cudaMalloc((void**)&foo, sizeof(int));
  cudaMemset(foo, 0, sizeof(int));
  printf("foo addr: 0x%x\n", (unsigned)(size_t)foo);

  diverge_cta<<<K, M*32>>>(foo);
  cudaMemcpy(&h_foo, foo, sizeof(int),cudaMemcpyDeviceToHost);
  if (h_foo == K*(M*N+5*(M-1))) {
    printf("Simple Scan Test PASSED\n");
  }
  else {
    printf("Result: %d\n", h_foo);
    printf("Simple Scan Test FAILED\n");
  }
  return 0;
}    

