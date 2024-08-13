#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>

#define BLOCKS 1
#define THREADS 1024



// Problemas a resolver:
// 1 - Maximo de threads por bloco: 1024
// 2 - Tamanho maximo da shared memory: 49152 -> Tamanho maximo de uint: 1536
int main(int argc, char** argv) {
  unsigned int realSize = 1024;
  unsigned int powerSize = getNextPowerOfTwo(realSize);
  unsigned int* array = generatePaddedArray(realSize, powerSize);

  //std::cout << "Input: " << powerSize << std::endl;
  //for (int i=0; i<powerSize ;i++)
  //  std::cout << array[i] << " ";
  //std::cout << std::endl;

  int *d_values;
  cudaMalloc((void**)&d_values, sizeof(int) * powerSize);
  cudaMemcpy(d_values, array, sizeof(int) * powerSize, cudaMemcpyHostToDevice);

  bitonicSort<<<BLOCKS, THREADS, THREADS * sizeof(unsigned int)>>>(d_values, powerSize);

  cudaMemcpy(array, d_values, sizeof(int) * powerSize, cudaMemcpyDeviceToHost);
  cudaFree(d_values);

  //std::cout << "Output: " << std::endl;
  //for (int i=0; i<powerSize ;i++)
  //  std::cout << array[i] << " ";
  //std::cout << std::endl;

  bool passed = true;
  for(int i = 1; i < realSize; i++)
    if (array[i-1] > array[i]) {
      passed = false;
      printf("Falha na posicao [%d] com [%d]\n", i-1, i);
    }

  printf("Test %s\n", passed ? "PASSED" : "FAILED");
}
