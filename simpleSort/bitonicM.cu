#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>

#include "simpleSort.cuh"
#include "bitonicM.cuh"



//--------------------------------------------------------------------------



// FOR INTERNAL USE
// funcao inline para simplificar nosso codigo
__device__ inline void swap(u_int &a, u_int &b) {
  u_int tmp = a;
  a = b;
  b = tmp;
}



//--------------------------------------------------------------------------



// FOR INTERNAL USE
// Function to get the next power of two greater than or equal to the given value
u_int getNextPowerOfTwo(size_t value) {
    u_int power = 1;
    while (power < value) { power <<= 1; }

    return power;
}


// Function to generate the padded array
// returns a pointer to a device memory of the array length
u_int* generatePaddedArray(u_int* d_array, size_t realSize, size_t nextPowerOfTwo) {
    // Ensure nextPowerOfTwo is a power of two
    if ((nextPowerOfTwo & (nextPowerOfTwo - 1)) != 0) {
      std::cerr << "nextPowerOfTwo must be a power of two." << std::endl;
      return nullptr;
    }

    //--

    // Allocate memory for the array
    u_int *d_arrayP2;
    cudaMalloc((void**)&d_arrayP2, sizeof(u_int) * nextPowerOfTwo);
    cudaMemset(d_arrayP2, 0, nextPowerOfTwo * sizeof(u_int));
    cudaMemcpy(d_arrayP2, d_array, sizeof(u_int) * realSize, cudaMemcpyDeviceToDevice);

    //--

    return d_arrayP2;
}


//--------------------------------------------------------------------------



// FOR INTERNAL USE
// funcao principal de bitonic sort
__global__ void bitonicSort(u_int *d_array, u_int size, u_int realSize) {
  extern __shared__ u_int shared[];

  // Get thread index within the block
  const u_int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Ensure we do not read out of bounds
  if (tid < size) {
    // Copy input to shared memory
    shared[threadIdx.x] = d_array[tid];
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
    d_array[tid] = shared[threadIdx.x];
  }
}



//--------------------------------------------------------------------------



// A CPU level function that preps the necessary variables for the
// bitonic sort, and then running it.
void B_bitonicProxy(u_int* d_array, u_int a_size) {
    u_int a_Pow2size = getNextPowerOfTwo(a_Pow2size);          // get next higher power of 2 size
    u_int* d_arrayP2 = generatePaddedArray(d_array, a_size, a_Pow2size);

    ////==== GET POWER OF 2 ARRAY

    bitonicSort<<<1, THREADS, SHAREDLIMIT>>>(d_arrayP2, a_Pow2size, a_size);

    ////==== CALCULATE BITONIC ON THIS POW2 ARRAY

    cudaMemcpy(d_array, d_arrayP2, sizeof(u_int) * a_size, cudaMemcpyDeviceToDevice);

    ////==== SEND BACK TO THE OTHER DEVICE MEMORY
}