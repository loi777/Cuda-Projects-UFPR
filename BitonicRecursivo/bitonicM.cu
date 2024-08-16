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
    // Allocate memory for the array
    u_int *d_arrayP2;
    cudaMalloc((void**)&d_arrayP2, sizeof(u_int) * nextPowerOfTwo);
    cudaMemset(d_arrayP2, __UINT32_MAX__, nextPowerOfTwo * sizeof(u_int));
    cudaMemcpy(d_arrayP2, d_array, sizeof(u_int) * realSize, cudaMemcpyDeviceToDevice);

    //--

    return d_arrayP2;
}


//--------------------------------------------------------------------------



// FOR INTERNAL USE
// funcao principal de bitonic sort
__global__ void bitonicSort(u_int *d_array, u_int size, int k, int j) {
  unsigned int i, ij;

    i = threadIdx.x + (blockDim.x * blockIdx.x);

    ij = i ^ j;

    if (i < size && ij < size) {

      if (ij > i) {                   // ij is to the right of i
        if ((i & k) == 0) {           // if the thread is going forward or back
          if (d_array[i] > d_array[ij]) {     // only invert  

            int temp = d_array[i];  // arr[i] receives arr[ij]
            d_array[i] = d_array[ij];
            d_array[ij] = temp;
          }
        } else {
          if (d_array[i] < d_array[ij]) {

            int temp = d_array[i];  // arr[i] receives arr[ij]
            d_array[i] = d_array[ij];
            d_array[ij] = temp;
          }
        }
      }

    }

    __syncthreads();
}



//--------------------------------------------------------------------------



// A CPU level function that preps the necessary variables for the
// bitonic sort, and then running it.
void  B_bitonicProxy(u_int* d_orig, u_int* d_array, u_int a_size) {
    u_int a_Pow2size = getNextPowerOfTwo(a_size);           // get next higher power of 2 size
    u_int* d_arrayP2 = generatePaddedArray(d_array, a_size, a_Pow2size);

    ////==== GET POWER OF 2 ARRAY

    // Perform Bitonic sort
    for (int k = 2; k <= a_Pow2size; k = k << 1) {
      // Bitonic merge
      for (int j = k / 2; j > 0; j = j >> 1) {
        bitonicSort<<<1, THREADS>>>(d_arrayP2, a_Pow2size, k, j);
      }
    }

    ////==== CALCULATE BITONIC ON THIS POW2 ARRAY

    cudaMemcpy(d_orig, d_arrayP2, sizeof(u_int) * a_size, cudaMemcpyDeviceToDevice);
    cudaFree(d_arrayP2);

    ////==== SEND BACK TO THE OTHER DEVICE MEMORY
}