#include <cuda_runtime.h>
#include <iostream>

#define NUM 16

#define BLOCKS 1
#define THREADS 1024

__device__ inline void swap(int & a, int & b) {
  // Alternative swap doesn't use a temporary register:
  a ^= b;
  b ^= a;
  a ^= b;
}

__global__ static void bitonicSort(int *values, int size) {
  extern __shared__ int shared[];
  const int tid = threadIdx.x;

  // Copy input to shared mem.
  shared[tid] = values[tid];
  __syncthreads();

  // Parallel bitonic sort.
  for (int k = 2; k <= size; k *= 2) {
    // Bitonic merge:
    for (int j = k / 2; j>0; j /= 2) {
      int ixj = tid ^ j;

      if (ixj > tid) {
        if ((tid & k) == 0) {
          if (shared[tid] > shared[ixj])
            swap(shared[tid], shared[ixj]);
        } else {
          if (shared[tid] < shared[ixj])
            swap(shared[tid], shared[ixj]);
        }
      }
      __syncthreads();
    }
  }

  // Write result.
  values[tid] = shared[tid];
}


// Coisas a resolver:
// 1 - Adicionar o padding para funcionar com qualquer tamanho
// 2 - Rodar com mais threads -> alterar shared memory
// 3 - Rodar com mais blocos  -> alterar shared memory
// 4 - Ordenar apenas uma parte do vetor
int main(int argc, char** argv) {
  int values[NUM];

  for(int i = 0; i < NUM; i++) { values[i] = rand(); }

  //std::cout << "Input: " << std::endl;
  //for (int i=0; i<NUM ;i++)
  //  std::cout << values[i] << " ";
  //std::cout << std::endl;

  int *d_values;
  cudaMalloc((void**)&d_values, sizeof(int) * NUM);
  cudaMemcpy(d_values, values, sizeof(int) * NUM, cudaMemcpyHostToDevice);

  bitonicSort<<<1, NUM, NUM * sizeof(int)>>>(d_values, NUM);

  cudaMemcpy(values, d_values, sizeof(int) * NUM, cudaMemcpyDeviceToHost);
  cudaFree(d_values);

  //std::cout << "Output: " << std::endl;
  //for (int i=0; i<NUM ;i++)
  //  std::cout << values[i] << " ";
  //std::cout << std::endl;

  bool passed = true;
  for(int i = 1; i < NUM; i++)
    if (values[i-1] > values[i])
      passed = false;

  printf("Test %s\n", passed ? "PASSED" : "FAILED");
}
