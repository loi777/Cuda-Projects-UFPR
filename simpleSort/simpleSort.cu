#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include <assert.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "chrono.c"

typedef unsigned int u_int;

#define NP 1             // Number of processors
#define BLOCKS 3         // Number of blocks per processor
#define THREADS 1024     // Number of threads per block

#define BINFIND(min, max, val, binSize, binQtd) (val >= max ? binQtd-1 : (val - min) / binSize)

// Enables maximum occupancy
#define SHARED_SIZE_LIMIT 1024U                       // bitonic sort lib

// Map to single instructions on G8x / G9x / G100
#define UMUL(a, b) __umul24((a), (b))                 // bitonic sort lib
#define UMAD(a, b, c) (UMUL((a), (b)) + (c))          // bitonic sort lib


//--------------------------------------------------------------------------



__device__ inline void Comparator(uint &keyA, uint &valA, uint &keyB,
                                  uint &valB, uint dir) {
  uint t;

  if ((keyA > keyB) == dir) {
    t = keyA;
    keyA = keyB;
    keyB = t;
    t = valA;
    valA = valB;
    valB = t;
  }
}



//--------------------------------------------------------------------------



// returns the size of the number group of each bin
// needs some strange calculations due to precision error
u_int getBinSize(u_int min, u_int max, int segCount) {
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


// Kernel para calcular histogramas em particoes
// Cada bloco eh responsavel por um histograma (linha da matriz)
__global__ void blockAndGlobalHisto(u_int *HH, u_int *Hg, u_int h, u_int *Input, u_int nElements, u_int nMin, u_int nMax, u_int segSize, u_int binWidth) {
    // Alloca shared memory para UM histograma
    extern __shared__ int sharedHist[];

    __syncthreads();  // threads will wait for the shared memory to be finished

    //---

    // Inicio da particao no vetor
    int blcStart = (blockIdx.x * segSize);    // bloco positionado na frente daquele que veio anterior a ele
    int thrPosi = threadIdx.x;              // 1 elemento por thread, starts as exactly the thread.x

    while(thrPosi < segSize && ((blcStart+thrPosi) < nElements)) {
        // Loop enquanto a thread estiver resolvendo elementos validos dentro do bloco e do array

        u_int val = Input[blcStart + thrPosi];    // get value
        int posi = BINFIND(nMin, nMax, val, binWidth, h);
        atomicAdd(&sharedHist[posi], 1);  // add to its corresponding segment
        atomicAdd(&Hg[posi], 1);  // add to its corresponding segment

        thrPosi += blockDim.x; // thread pula para frente, garantindo que nao ira processar um valor ja processado
        // saira do bloco quando terminar todos os pixeis dos quais eh responsavel
    }

    __syncthreads();

    //--

    // Passa os resultados da shared memory para matriz
    // deixar isso a cargo da thread 0 eh mais modular que mandar uma pra uma
    if (threadIdx.x == 0) {
      for (int i = 0; i < h; i++) {
        atomicAdd(&HH[(blockIdx.x * h) + i], sharedHist[i]); 
      }
    }
    // Y: (blockIdx.x * segCount)
    // X: threadIdx.x

    __syncthreads();
}


// calculates the scan of the global histogram and saves it into the horizontal scan
__global__ void globalHistoScan(u_int *Hg, u_int *SHg, u_int h){
    // Obtem shared memory para o histogram horizontal
    extern __shared__ u_int _SHg[];

    //--

    int threadPosi = threadIdx.x;         // starts as thread ID

    //--

    while(threadPosi < h) {
      // Loop while inside the histogram

      int sum = 0;

      for (int i = threadPosi-1; i >= 0; i--) {
        // makes the individual sum of every index before this one
        sum += Hg[i];
      }

      _SHg[threadPosi] = sum;

      //--

      threadPosi += blockDim.x; // go to the next element
    }

    __syncthreads();

    //--

    // Passa os resultados da shared memory para o scan
    // deixar isso a cargo da thread 0 eh mais modular que mandar uma pra uma
    if (threadIdx.x == 0) {
      for (int i = 0; i < h; i++) {
        SHg[i] = _SHg[i];
      }
    }

    __syncthreads();
}


// calculates the scan of each non-global histogram, saving it in different lines of the vertical scan
__global__ void verticalScanHH(u_int *Hg, u_int *PSv, u_int* PSh, u_int h, u_int hist_count){
    // Obtem shared memory para o histogram horizontal
    extern __shared__ u_int _PSv[];

    //--
    
    int posiX = threadIdx.x;         // starts as thread ID

    //--

    while(posiX < h) {
      // Loop while inside the histogram's segments

      int sum = 0;

      for (int posiY = 0; posiY < hist_count; posiY++) {
        _PSv[posiX + (posiY*h)] = sum + PSh[posiX];
        sum += Hg[posiX + (posiY*h)];
      }

      //--

      posiX += blockDim.x;
      // jumps to the next unprocessed column
    }

    __syncthreads();

    //--

    // Passa os resultados da shared memory para o scan
    // deixar isso a cargo da thread 0 eh mais modular que mandar uma pra uma
    if (threadIdx.x == 0) {
      for (int i = 0; i < h; i++) {
        PSv[i] = _PSv[i];
      }
    }

    __syncthreads();
}


// Uses the consultation table to separate the groups of numbers according to their bins
// saves in output device memory
__global__ void PartitionKernel(u_int* output, u_int* input, u_int* table, int arraySize, int segSize, int segCount, u_int minVal, u_int maxVal, u_int binWidth) {
  int posiX = threadIdx.x;
  int blkDiff = (blockIdx.x * segSize);

  //--

  while((posiX < segSize) && ((posiX+blkDiff) < arraySize)) {
    // while inside the block scope and inside the array

    //                     X                                                                Y
    int tableID = BINFIND(minVal, maxVal, input[posiX+blkDiff], binWidth, segCount) + (blockIdx.x*segCount);
    int posi = atomicAdd(&table[tableID], 1);

    output[posi] = input[posiX+blkDiff];

    // jumps to next unprocessed element
    posiX += blockDim.x;
  }

  //--

  __syncthreads();
}


// Combined bitonic merge steps for
// size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__ void bitonicMergeShared(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint arrayLength, uint size, uint dir) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  //--

  // Shared memory storage for current subarray
  __shared__ uint s_key[SHARED_SIZE_LIMIT];
  __shared__ uint s_val[SHARED_SIZE_LIMIT];

  d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  s_key[threadIdx.x + 0] = d_SrcKey[0];
  s_val[threadIdx.x + 0] = d_SrcVal[0];
  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
  s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

  //--

  // Bitonic merge
  uint comparatorI = UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
  uint ddd = dir ^ ((comparatorI & (size / 2)) != 0); // xor

  for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
    cg::sync(cta);
    uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
  }

  //--

  cg::sync(cta);
  d_DstKey[0] = s_key[threadIdx.x + 0];
  d_DstVal[0] = s_val[threadIdx.x + 0];
  d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
  d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}



//---------------------------------------------------------------------------------


//cudaMemcpy(Output, d_Output, nTotalElements * sizeof(u_int), cudaMemcpyDeviceToHost);
//verifySort(Input, Output, nTotalElements);
void verifySort(u_int *Input, u_int *Output, u_int nElements) {
  thrust::device_vector<u_int> d_Input(Input, Input + nElements);
  thrust::device_vector<u_int> d_Output(Output, Output + nElements);
  thrust::sort(d_Input.begin(), d_Input.end());

  bool isSorted = thrust::equal(d_Input.begin(), d_Input.end(), d_Output.begin());
  if (isSorted) { std::cout << "Sort verification: SUCCESS" << std::endl; } 
  else          { std::cout << "Sort verification: FAILURE" << std::endl; }
}


//---------------------------------------------------------------------------------


// Create and generate a random array of nElements
// returns as a pointer
u_int* genRandomArray(int nElem) {
  u_int* array = new u_int[nElem];

  for (int i = 0; i < nElem; ++i) {
    int a = std::rand() % 50;
    int b = std::rand();
    u_int v = a * 100 + b;
    array[i] = v;
  }

  return array;
}


//---------------------------------------------------------------------------------


u_int check_parameters(int argc){
  if (argc != 4) {
    std::cerr << "Usage: ./simpleSort <nTotalElements> <h> <nR>" << std::endl;
    return EXIT_FAILURE;
  }
  return 0;
}


//---------------------------------------------------------------------------------


int main(int argc, char* argv[]) {
  if (check_parameters(argc)) { return EXIT_FAILURE; }
  std::srand(std::time(nullptr));

  //u_int nTotalElements = std::stoi(argv[1]);                    // Numero de elementos
  //u_int h = std::stoi(argv[2]);                                 // Numero de histogramas
  //u_int nR = std::stoi(argv[3]);                                // Numero de chamadas do kernel
  //u_int *Input = genRandomArray(nTotalElements);                // Vetor de entrada
  u_int nTotalElements = 18;
  u_int h = 6;
  u_int nR = 1;
  u_int Input[] = {2, 4, 33, 27, 8, 10, 42, 3, 12, 21, 10, 12, 15, 27, 38, 45, 18, 22};
  u_int *Output = new u_int[nTotalElements];                      // Vetor ordenado
  u_int SEG_SIZE = (ceil((float)nTotalElements/((float)NP*(float)BLOCKS)));
  chronometer_t chrono_Thrust, chrono_Hist;

  // Busca menor valor, maior valor e o comprimento do bin
  u_int nMin = *std::min_element(Input, Input + nTotalElements);
  u_int nMax = *std::max_element(Input, Input + nTotalElements);
  u_int binWidth = getBinSize(nMin, nMax, h);

  // Information printing, a pedido do Zola
  std::cout << "Min: " << nMin << " | Max: " << nMax << std::endl;
  std::cout << "Largura da Faixa: " << binWidth << std::endl;

  // Aloca cores e copia para GPU
  unsigned int *d_Input, *d_Output, *HH, *Hg, *SHg, *PSv, *V;
  cudaMalloc((void**)&d_Input,  nTotalElements  * sizeof(u_int));  // device input data
  cudaMalloc((void**)&d_Output, nTotalElements  * sizeof(u_int));  // device input sorted data
  cudaMalloc((void**)&HH,       NP * BLOCKS * h * sizeof(u_int));  // device histogram matrix
  cudaMalloc((void**)&Hg,       h               * sizeof(u_int));  // device histogram sum
  cudaMalloc((void**)&SHg,      h               * sizeof(u_int));  // device histogram prefix sum
  cudaMalloc((void**)&PSv,      NP * BLOCKS * h * sizeof(u_int));  // device matrix vertical prefix sum
  cudaMalloc((void**)&V,        NP * BLOCKS * h * sizeof(u_int));
  cudaMemcpy(d_Input, Input, nTotalElements * sizeof(u_int), cudaMemcpyHostToDevice);

  chrono_reset(&chrono_Thrust);
  chrono_reset(&chrono_Hist);

  for (int i = 0; i < nR; ++i) {
    chrono_start(&chrono_Hist);
    blockAndGlobalHisto<<<NP*BLOCKS, THREADS, SEG_SIZE>>>(HH, Hg, h, d_Input, nTotalElements, nMin, nMax, SEG_SIZE, binWidth);
    globalHistoScan<<<1, THREADS, h>>>(Hg, SHg, h);
    verticalScanHH<<<1, THREADS, h>>>(HH, PSv, SHg, h, NP*BLOCKS);
    PartitionKernel<<<NP*BLOCKS, THREADS>>>(d_Output, d_Input, V, nTotalElements, SEG_SIZE, h, nMin, nMax, binWidth);
    bitonicSort();
    chrono_stop(&chrono_Hist);

    cudaMemcpy(Output, d_Output, nTotalElements * sizeof(u_int), cudaMemcpyDeviceToHost);
    thrust::device_vector<u_int> th_Input(Input, Input + nTotalElements);
    thrust::device_vector<u_int> th_Output(Output, Output + nTotalElements);
    
    chrono_start(&chrono_Thrust);
    thrust::sort(th_Input.begin(), th_Input.end());
    chrono_stop(&chrono_Thrust);

    bool isSorted = thrust::equal(th_Input.begin(), th_Input.end(), th_Output.begin());
    if (isSorted) { std::cout << "Sort " << i << " verification: SUCCESS" << std::endl; } 
    else          { std::cout << "Sort " << i << " verification: FAILURE" << std::endl; }
  }

  // ---

  printf("\n----THRUST\n");
  printf("Delta time: " );
  chrono_report_TimeInLoop( &chrono_Thrust, (char *)"thrust max_element", nR);

  double thrust_time_seconds = (double) chrono_gettotal( &chrono_Thrust )/((double)1000*1000*1000);
  printf( "Tempo em segundos: %lf s\n", thrust_time_seconds );
  printf( "Vazão: %lf INT/s\n", (nTotalElements)/thrust_time_seconds );
  
  //--

  printf("\n----HISTOGRAM\n");
  printf("Delta time: " );
  chrono_report_TimeInLoop( &chrono_Hist, (char *)"reduceMax_persist", nR);

  double reduce_time_seconds = (double) chrono_gettotal( &chrono_Hist )/((double)1000*1000*1000);
  printf( "Tempo em segundos: %lf s\n", reduce_time_seconds );
  printf( "Vazão: %lf INT/s\n", (nTotalElements)/reduce_time_seconds );

  printf("\n--Tempo em relacao ao Thrust\n");
  printf("Em segundos: %lf\n", reduce_time_seconds - thrust_time_seconds);
  printf("Em porcento: %d\n", (int)((thrust_time_seconds/reduce_time_seconds)*100.0));

  //--

  cudaFree(d_Input);
  cudaFree(d_Output);
  cudaFree(HH);
  cudaFree(Hg);
  cudaFree(SHg);
  cudaFree(PSv);
  cudaFree(V);

  //delete[] Input;
  delete[] Output;

  return EXIT_SUCCESS;
}

