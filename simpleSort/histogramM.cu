#include <cuda.h>

#include "simpleSort.cuh"
#include "histogramM.cuh"

#define BINSTART(min, binSize, i) ((u_int)((binWidth*i)+min))
#define BINEND(min, binSize, i) ((u_int)(BINSTART(min, binSize, (i+1))-1))
#define BINFIND(min, max, val, binSize, binQtd) (val >= max ? binQtd-1 : (val - min) / binSize)



//--------------------------------------------------------------------------



// returns the size of the number group of each bin
// needs some strange calculations due to precision error
u_int H_getBinSize(u_int min, u_int max, int segCount) {
  u_int binSize = max - min;
  if ((binSize % segCount) == 0) {
    // complete division
    binSize /= segCount;
  } else {
    // incomplete division
    binSize /= segCount;
    binSize++;
  }

  return binSize;
}



//---------------------------------------------------------------------------------



// return the min of a device array
__global__ u_int H_getMin(u_int* d_array, u_int size) {
    __shared__ int min;

    u_int posi = (blockIdx.x*blockDim.x) + threadIdx.x;

    while(posi < size) {

        min = atomicMin(&d_array[posi], min);

        posi += blockDim.x;
    }

    __syncthreads();

    //--

    return min;
}


// return the max of a device array
__global__ u_int H_getMax(u_int* d_array, u_int size) {
    __shared__ int max;

    u_int posi = (blockIdx.x*blockDim.x) + threadIdx.x;

    while(posi < size) {

        max = atomicMax(&d_array[posi], max);

        posi += blockDim.x;
    }

    __syncthreads();

    //--

    return max;
}



//--------------------------------------------------------------------------


// Kernel para calcular histogramas em particoes
// Cada bloco eh responsavel por um histograma (linha da matriz)
__global__ void H_getHistogram(u_int *HH, u_int *Hg, u_int h, u_int *Input, u_int nElements, u_int nMin, u_int nMax, u_int segSize, u_int binWidth) {
    // Alloca shared memory para UM histograma
    extern __shared__ int _HH[];
    if (threadIdx.x < h) { _HH[threadIdx.x] = 0; }
    __syncthreads();

    //---

    // Inicio da particao no vetor
    int blcStart = (blockIdx.x * segSize);    // bloco positionado na frente daquele que veio anterior a ele
    int thrdPosi = threadIdx.x;              // 1 elemento por thread, starts as exactly the thread.x

    while(thrdPosi < segSize && ((blcStart+thrdPosi) < nElements)) {
        // Loop enquanto a thread estiver resolvendo elementos validos dentro do bloco e do array
        u_int val = Input[blcStart + thrdPosi];    // get value
        int posi = BINFIND(nMin, nMax, val, binWidth, h);
        atomicAdd(&_HH[posi], 1);  // add to its corresponding segment
        atomicAdd(&Hg[posi], 1);  // add to its corresponding segment

        thrdPosi += blockDim.x; // thread pula para frente, garantindo que nao ira processar um valor ja processado
    }

    __syncthreads();

    //--

    // Passa os resultados da shared memory para matriz
    // deixar isso a cargo da thread 0 eh mais modular que mandar uma pra uma
    if (threadIdx.x < h)
      atomicAdd(&HH[(blockIdx.x * h) + threadIdx.x], _HH[threadIdx.x]);
    __syncthreads();
}


// calculates the scan of the global histogram and saves it into the horizontal scan
__global__ void H_horizontalScan(u_int *Hg, u_int *SHg, u_int h) {
    // Obtem shared memory para o histogram horizontal
    extern __shared__ u_int _SHg[];
    if (threadIdx.x < h) { _SHg[threadIdx.x] = 0; }
    __syncthreads();

    //--

    u_int thrdPosi = threadIdx.x;         // starts as thread ID

    //--

    while (thrdPosi < h) {
      // Loop while inside the histogram
      u_int sum = 0;
      for (int i = thrdPosi-1; i >= 0; i--) {
        sum += Hg[i]; // makes the individual sum of every index before this one
      }
      _SHg[thrdPosi] = sum;

      //--

      thrdPosi += blockDim.x; // go to the next element
    }

    __syncthreads();

    //--

    // Passa os resultados da shared memory para o scan
    // deixar isso a cargo da thread 0 eh mais modular que mandar uma pra uma
    if (threadIdx.x < h)
      SHg[threadIdx.x] = _SHg[threadIdx.x];
    __syncthreads();
}


// calculates the scan of each non-global histogram, saving it in different lines of the vertical scan
__global__ void H_verticalScan(u_int *HH, u_int *PSv, u_int h) {
    // Obtem shared memory para o histogram horizontal
    extern __shared__ u_int _PSv[];
    if (threadIdx.x < h) { _PSv[threadIdx.x] = 0; }
    __syncthreads();

    //--

    u_int thrdPosi = threadIdx.x;     // Thread por coluna

    //--

    while (thrdPosi < h) {
      int sum = 0;
      for (int i=blockIdx.x-1; i>=0; i--){
        sum += HH[i*h + thrdPosi];
      }
      _PSv[thrdPosi] = sum;

      thrdPosi += blockDim.x; // go to the next element
    }
    __syncthreads();

    //--

    // Passa os resultados da shared memory para o scan
    // deixar isso a cargo da thread 0 eh mais modular que mandar uma pra uma
    if (threadIdx.x < h)
      PSv[blockIdx.x*h + threadIdx.x] = _PSv[threadIdx.x];
    __syncthreads();
}



//--------------------------------------------------------------------------



// Uses the consultation table to separate the groups of numbers according to their bins
// saves in output device memory
__global__ void H_Partitioner(u_int *HH, u_int *SHg, u_int *PSv, u_int h, u_int *Input, u_int *Output, u_int nElements, u_int nMin, u_int nMax, u_int segSize, u_int binWidth) {
    extern __shared__ u_int _HLsh[];
    if (threadIdx.x < h) { _HLsh[threadIdx.x] = 0; }
    __syncthreads();

    // Thread ID and total threads
    u_int thrdPosi = threadIdx.x; 
    u_int totalThreads = blockDim.x;

    // Calculate the indices for shared memory
    while (thrdPosi < h) {
        _HLsh[thrdPosi] = SHg[thrdPosi] + PSv[blockIdx.x * h + thrdPosi];
        thrdPosi += totalThreads;
    }
    __syncthreads();

    // Reset thread position for the next phase
    thrdPosi = threadIdx.x;

    // Process elements in the segment
    while (thrdPosi < segSize && ((blockIdx.x * segSize + thrdPosi) < nElements)) {
        u_int val = Input[blockIdx.x * segSize + thrdPosi]; 
        u_int posi = BINFIND(nMin, nMax, val, binWidth, h);

        // Atomic operation to update the output array
        u_int index = atomicAdd(&_HLsh[posi], 1);    // Get the current position and increment it atomically
        if (index < nElements)
          Output[index] = val;                         // Write the value to the output array

        thrdPosi += totalThreads;
    }
    __syncthreads();
}