#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <iostream>
#include <limits>

#include "sortingNetworks_common.h"

#define N 33554432  // Ordenacao maxima do bitonic sort

// Function to get the next power of two greater than or equal to the given value
unsigned int getNextPowerOfTwo(size_t value) {
  unsigned int power = 1;
  while (power < value) { power <<= 1; }

  return power;
}

// Function to generate the padded array
void generatePaddedArray(unsigned int *array, unsigned int *index, size_t realSize, size_t nextPowerOfTwo) {
  // Fill the array with random values for the first part
  for (size_t i = 0; i < realSize; ++i) { 
    array[i] = std::rand() % 10; 
    index[i] = i;
  }
  // Fill the rest of the array with the maximum unsigned int value
  for (size_t i = realSize; i < nextPowerOfTwo; ++i) { 
    array[i] = UINT_MAX; 
    index[i] = i;
  }
}

int main(int argc, char** argv) {
  std::srand(std::time(nullptr));
  unsigned int Size = 16000000;
  unsigned int powerSize = getNextPowerOfTwo(Size);

  // Alocate host and CUDA vectors
  uint *h_Input, *h_InputIdx, *h_Output, *h_OutputIdx;
  uint *d_Input, *d_InputIdx, *d_Output, *d_OutputIdx;
  h_Input     = (uint *)malloc(powerSize * sizeof(uint));
  h_InputIdx  = (uint *)malloc(powerSize * sizeof(uint));
  h_Output    = (uint *)malloc(powerSize * sizeof(uint));
  h_OutputIdx = (uint *)malloc(powerSize * sizeof(uint));
  cudaMalloc((void **)&d_Input,     N * sizeof(uint)); // Tamanho deve ser N por conta do bitonic
  cudaMalloc((void **)&d_InputIdx,  N * sizeof(uint)); // Tamanho deve ser N por conta do bitonic
  cudaMalloc((void **)&d_Output,    N * sizeof(uint)); // Tamanho deve ser N por conta do bitonic
  cudaMalloc((void **)&d_OutputIdx, N * sizeof(uint)); // Tamanho deve ser N por conta do bitonic

  // Insert random elements
  generatePaddedArray(h_Input, h_InputIdx, Size, powerSize);

  // Copy memory to device
  cudaMemcpy(d_Input,    h_Input,    powerSize * sizeof(uint), cudaMemcpyHostToDevice);
  cudaMemcpy(d_InputIdx, h_InputIdx, powerSize * sizeof(uint), cudaMemcpyHostToDevice);

  //std::cout << "Input: " << powerSize << std::endl;
  //for (int i=0; i<powerSize ;i++) { std::cout << h_Input[i] << " "; }
  //std::cout << std::endl;

  bitonicSort(d_Output, d_OutputIdx, d_Input, d_InputIdx, N / powerSize, powerSize, 0);

  cudaMemcpy(h_Output, d_Output, powerSize * sizeof(uint), cudaMemcpyDeviceToHost);

  //std::cout << "Output: " << std::endl;
  //for (int i=0; i<powerSize ;i++) { std::cout << h_Output[i] << " "; }
  //std::cout << std::endl;

  bool passed = true;
  for(int i = 1; i < Size; i++)
    if (h_Output[i-1] > h_Output[i]) {
      passed = false;
      printf("Falha na posicao [%d] com [%d]\n", i-1, i);
    }
  printf("Sort of %d values: %s\n", Size, passed ? "PASSED" : "FAILED");

  cudaFree(d_Input);
  cudaFree(d_InputIdx);
  cudaFree(d_Output);
  cudaFree(d_OutputIdx);
  free(h_Input);
  free(h_InputIdx);
  free(h_Output);
  free(h_OutputIdx);
}
