#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>

#include "simpleSort.cuh"
#include "bitonicM.cuh"



//--------------------------------------------------------------------------



// For internal use
// Function to get the next power of two greater than or equal to the given value
unsigned int getNextPowerOfTwo(size_t value) {
  unsigned int power = 1;
  while (power < value) { power <<= 1; }

  return power;
}


// For internal use
// funcao inline para simplificar nosso codigo
__device__ inline void swap(int &a, int &b) {
  int tmp = a;
  a = b;
  b = tmp;
}



//--------------------------------------------------------------------------



// Function to generate the padded array
unsigned int* generatePaddedArray(size_t realSize, size_t nextPowerOfTwo) {
  // Ensure nextPowerOfTwo is a power of two
  if ((nextPowerOfTwo & (nextPowerOfTwo - 1)) != 0) {
    std::cerr << "nextPowerOfTwo must be a power of two." << std::endl;
    return nullptr;
  }

  // Allocate memory for the array
  unsigned int *array = new unsigned int[nextPowerOfTwo];
  // Fill the array with random values for the first part
  for (size_t i = 0; i < realSize; ++i) { array[i] = std::rand(); }
  // Fill the rest of the array with the maximum unsigned int value
  for (size_t i = realSize; i < nextPowerOfTwo; ++i) { array[i] = UINT_MAX; }

  return array;
}



//--------------------------------------------------------------------------



// funcao principal de bitonic sort
__global__ void bitonicSort(int *values, int size) {
  extern __shared__ int shared[];

  // Get thread index within the block
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Ensure we do not read out of bounds
  if (tid < size) {
    // Copy input to shared memory
    shared[threadIdx.x] = values[tid];
    __syncthreads();

    // Perform Bitonic sort
    for (int k = 2; k <= size; k *= 2) {
      // Bitonic merge
      for (int j = k / 2; j > 0; j /= 2) {
        int ixj = threadIdx.x ^ j;

        // Ensure ixj is within bounds
        if (ixj > threadIdx.x && ixj < blockDim.x) {
          if ((threadIdx.x & k) == 0) {
            if (shared[threadIdx.x] > shared[ixj]) { swap(shared[threadIdx.x], shared[ixj]); }
          } else {
            if (shared[threadIdx.x] < shared[ixj]) { swap(shared[threadIdx.x], shared[ixj]); }
          }
        }
        __syncthreads();
      }
    }

    // Write the result back to global memory
    values[tid] = shared[threadIdx.x];
  }
}