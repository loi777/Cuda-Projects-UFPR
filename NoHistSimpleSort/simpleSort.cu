// AUTORES:
// Muriki Gusmão Yamanaka - GRR20203933
// Luiz Fernando Giongo Dos Santos - GRR20203965

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "sortingNetworks_common.h"
#include "chrono.c"

typedef unsigned int u_int;

#define NP 2             // Number of processors
#define BLOCKS 28        // Number of blocks per processor
#define NB NP*BLOCKS     // Total number of blocks
#define THREADS 1024     // Number of threads per block

#define N 33554432       // Tamanho maximo do bitonic sort

#define BINFIND(min, max, val, binSize, binQtd) (val >= max ? binQtd-1 : (val - min) / binSize)

//--------------------------------------------------------------------------

// returns the size of the number group of each bin
// needs some strange calculations due to precision error
u_int getBinSize(u_int min, u_int max, u_int segCount) {
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
    extern __shared__ u_int _HH[];
    if (threadIdx.x < h) { _HH[threadIdx.x] = 0; }
    __syncthreads();

    //---

    // Inicio da particao no vetor
    u_int blcStart = (blockIdx.x * segSize);    // bloco positionado na frente daquele que veio anterior a ele
    u_int thrdPosi = threadIdx.x;              // 1 elemento por thread, starts as exactly the thread.x

    while(thrdPosi < segSize && ((blcStart+thrdPosi) < nElements)) {
        // Loop enquanto a thread estiver resolvendo elementos validos dentro do bloco e do array
        u_int val = Input[blcStart + thrdPosi];    // get value
        u_int posi = BINFIND(nMin, nMax, val, binWidth, h);
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
__global__ void globalHistoScan(u_int *Hg, u_int *SHg, u_int h){
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
__global__ void verticalScanHH(u_int *HH, u_int *PSv, u_int h){
    // Obtem shared memory para o histogram horizontal
    extern __shared__ u_int _PSv[];
    if (threadIdx.x < h) { _PSv[threadIdx.x] = 0; }
    __syncthreads();

    //--

    u_int thrdPosi = threadIdx.x;     // Thread por coluna

    //--

    while (thrdPosi < h) {
      u_int sum = 0;
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


// Uses the consultation table to separate the groups of numbers according to their bins
// saves in output device memory
__global__ void PartitionKernel(u_int *HH, u_int *SHg, u_int *PSv, u_int h, u_int *Input, u_int *Output, u_int nElements, u_int nMin, u_int nMax, u_int segSize, u_int binWidth) {
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


void thrustSortProxy(u_int* h_array, u_int start, u_int end) {
  thrust::device_vector<u_int> d_vec(&h_array[start], &h_array[start] + (end-start));

  thrust::sort(d_vec.begin(), d_vec.end());

  thrust::copy(d_vec.begin(), d_vec.end(), &h_array[start]);
}


//---------------------------------------------------------------------------------


//cudaMemcpy(Output, d_Output, nTotalElements * sizeof(u_int), cudaMemcpyDeviceToHost);
//verifySort(Input, Output, nTotalElements);
void verifySort(u_int *Input, u_int *Output, u_int nElements, chronometer_t *chrono_Thrust, u_int k) {
  thrust::device_vector<u_int> th_Input(Input, Input + nElements);
  thrust::device_vector<u_int> th_Output(Output, Output + nElements);
  
  chrono_start(chrono_Thrust);
  thrust::sort(th_Input.begin(), th_Input.end());
  chrono_stop(chrono_Thrust);

  bool isSorted = thrust::equal(th_Input.begin(), th_Input.end(), th_Output.begin());
  if (isSorted) { std::cout << "Sort " << k << " verification: SUCCESS" << std::endl; } 
  else          { std::cout << "Sort " << k << " verification: FAILURE" << std::endl; }
}


//---------------------------------------------------------------------------------


// Create and generate a random array of nElements
// returns as a pointer
u_int* genRandomArray(int nElem) {
  u_int* array = new u_int[nElem];

  for (int i = 0; i < nElem; ++i) {
    u_int a = std::rand() % 50;
    u_int b = std::rand();
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

void cudaResetVariables(u_int *HH, u_int *Hg, u_int *SHg, u_int *PSv, u_int h){
  cudaMemset(HH,  0, NB * h * sizeof(u_int));
  cudaMemset(PSv, 0, NB * h * sizeof(u_int));
  cudaMemset(Hg,  0, h * sizeof(u_int));
  cudaMemset(SHg, 0, h * sizeof(u_int));
}


//---------------------------------------------------------------------------------

// Function to get the next power of two greater than or equal to the given value
u_int getNextPowerOfTwo(size_t value) {
  u_int power = 1;
  while (power < value) { power <<= 1; }

  return power;
}

// Function to generate the padded array
void generatePaddedArray(u_int *array, u_int *index, size_t realSize, size_t nextPowerOfTwo) {
  // Fill the array with random values for the first part
  for (size_t i = 0; i < realSize; ++i) { 
    array[i] = std::rand(); 
    index[i] = i;
  }
  // Fill the rest of the array with the maximum unsigned int value
  for (size_t i = realSize; i < nextPowerOfTwo; ++i) { 
    array[i] = UINT_MAX; 
    index[i] = i;
  }
}

int main(int argc, char* argv[]) {
  if (check_parameters(argc)) { return EXIT_FAILURE; }
  std::srand(std::time(nullptr));

  u_int nTotalElements = std::stoi(argv[1]);                    // Numero de elementos
  u_int powerElements = getNextPowerOfTwo(nTotalElements);      // Numero de elementos
  u_int h = std::stoi(argv[2]);                                 // Numero de histogramas
  u_int nR = std::stoi(argv[3]);                                // Numero de chamadas do kernel
  u_int *Input  = new u_int[powerElements];                    // Vetor de entrada
  u_int *InputIdx  = new u_int[powerElements];                 // Vetor indexado
  u_int *Output = new u_int[powerElements];                    // Vetor ordenado
  u_int *OutputIdx = new u_int[powerElements];                 // Vetor indexado
  u_int *h_SHg  = new u_int[h];                                 // host soma de prefixos
  u_int SEG_SIZE = (ceil((float)nTotalElements/((float)NB)));
  chronometer_t chrono_Thrust, chrono_Hist;

  // Gera dados
  generatePaddedArray(Input, InputIdx, nTotalElements, powerElements);

  // Busca menor valor, maior valor e o comprimento do bin
  u_int nMin = *std::min_element(Input, Input + nTotalElements);
  u_int nMax = *std::max_element(Input, Input + nTotalElements);
  u_int binWidth = getBinSize(nMin, nMax, h);

  // Information printing, a pedido do Zola
  std::cout << "Min: " << nMin << " | Max: " << nMax << std::endl;
  std::cout << "Largura da Faixa: " << binWidth << std::endl;

  // Aloca cores e copia para GPU
  u_int *d_Input, *d_InputIdx;
  u_int *d_Output,*d_OutputIdx;
  u_int *HH, *Hg, *SHg, *PSv;
  cudaMalloc((void**)&d_Input,     N * sizeof(u_int));  // device input data
  cudaMalloc((void**)&d_InputIdx,  N * sizeof(u_int));  // device input data
  cudaMalloc((void**)&d_Output,    N * sizeof(u_int));  // device input sorted data
  cudaMalloc((void**)&d_OutputIdx, N * sizeof(u_int));  // device input sorted data
  cudaMalloc((void**)&HH,          NB * h         * sizeof(u_int));  // device histogram matrix
  cudaMalloc((void**)&Hg,          h              * sizeof(u_int));  // device histogram sum
  cudaMalloc((void**)&SHg,         h              * sizeof(u_int));  // device histogram prefix sum
  cudaMalloc((void**)&PSv,         NB * h         * sizeof(u_int));  // device matrix vertical prefix sum

  //std::cout << "Input: " << std::endl;
  //for (size_t i=0; i<powerElements ;i++) std::cout << Input[i] << " ";
  //std::cout << std::endl;

  // Copia memoria
  cudaMemcpy(d_Input, Input, powerElements * sizeof(u_int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_InputIdx, InputIdx, powerElements * sizeof(u_int), cudaMemcpyHostToDevice);

  chrono_reset(&chrono_Thrust);
  chrono_reset(&chrono_Hist);

  for (int k = 0; k < nR; k++) {
    cudaResetVariables(HH, Hg, SHg, PSv, h);
    // --
    chrono_start(&chrono_Hist);
    //blockAndGlobalHisto<<<NB, THREADS, h*sizeof(u_int)>>>(HH, Hg, h, d_Input, nTotalElements, nMin, nMax, SEG_SIZE, binWidth);
    //globalHistoScan    <<<1,  THREADS, h*sizeof(u_int)>>>(Hg, SHg, h);
    //verticalScanHH     <<<NB, THREADS, h*sizeof(u_int)>>>(HH, PSv, h);
    //PartitionKernel    <<<NB, THREADS, h*sizeof(u_int)>>>(HH, SHg, PSv, h, d_Input, d_Output, nTotalElements, nMin, nMax, SEG_SIZE, binWidth);
    //cudaMemcpy(h_SHg, SHg, h * sizeof(u_int), cudaMemcpyDeviceToHost);

    bitonicSort(d_Output, d_OutputIdx, d_Input, d_InputIdx, N / powerElements, powerElements, 1);
    chrono_stop(&chrono_Hist);
    // --
    cudaMemcpy(Output, d_Output, powerElements * sizeof(u_int), cudaMemcpyDeviceToHost);
    // --
    verifySort(Input, Output, nTotalElements, &chrono_Thrust, k);
  }

  //std::cout << "Output: " << std::endl;
  //for (size_t i=0; i<powerElements ;i++) std::cout << Output[i] << " ";
  //std::cout << std::endl;

  // ---

  printf("\n----THRUST\n");
  printf("Delta time: " );
  chrono_report_TimeInLoop( &chrono_Thrust, (char *)"thrust sort", nR);

  double thrust_time_seconds = (double) chrono_gettotal( &chrono_Thrust )/((double)1000*1000*1000);
  printf( "Tempo em milisegundos: %lf ms\n", thrust_time_seconds*1000 );
  printf( "Vazão: %lf GINT/s\n", (nTotalElements)/(thrust_time_seconds*1000000000) );
  
  //--

  printf("\n----HISTOGRAM\n");
  printf("Delta time: " );
  chrono_report_TimeInLoop( &chrono_Hist, (char *)"histogram sort", nR);

  double reduce_time_seconds = (double) chrono_gettotal( &chrono_Hist )/((double)1000*1000*1000);
  printf( "Tempo em milisegundos: %lf ms\n", reduce_time_seconds*1000 );
  printf( "Vazão: %lf GINT/s\n", (nTotalElements)/(reduce_time_seconds*1000000000) );

  printf("\n--Tempo em relacao ao Thrust\n");
  printf("Em milisegundos: %lf\n", (reduce_time_seconds - thrust_time_seconds)*1000);
  printf("Em porcento: %d\n", (int)((thrust_time_seconds/reduce_time_seconds)*100.0));

  //--

  cudaFree(d_Input);
  cudaFree(d_Output);
  cudaFree(HH);
  cudaFree(Hg);
  cudaFree(SHg);
  cudaFree(PSv);

  //delete[] Input;
  delete[] Output;

  return EXIT_SUCCESS;
}

    // launch kernel that that sorts the inside of each bin partition
    //cudaMemcpy(h_SHg, SHg, h * sizeof(u_int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Output, d_Output, nTotalElements * sizeof(u_int), cudaMemcpyDeviceToHost);
    //u_int start, end = 0;
    //for (u_int bin = 1; bin < h; bin++) {
    //  // call a bitonic sort for every bin
    //  start = end;
    //  end = h_SHg[bin];
    //  thrustSortProxy(Output, start, end);
    //}
    //start = end;
    //end = nTotalElements;
    //thrustSortProxy(Output, start, end);
