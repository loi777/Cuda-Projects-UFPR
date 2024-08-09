#include <iostream>
#include <sys/types.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

typedef unsigned int u_int;

#define NP 1             // Number of processors
#define BLOCKS 3         // Number of blocks per processor
#define THREADS 1024     // Number of threads per block

#define BINFIND(min, max, val, binSize, binQtd) (val >= max ? binQtd-1 : (val - min) / binSize)

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
    int threadPosi = threadIdx.x;         // starts as thread ID

    //--

    while(threadPosi < h) {
      // Loop while inside the histogram

      int sum = 0;

      for (int i = threadPosi-1; i >= 0; i--) {
        // makes the individual sum of every index before this one
        sum += Hg[i];
      }

      SHg[threadPosi] = sum;

      //--

      threadPosi += blockDim.x; // go to the next element
    }

    //--

    __syncthreads();
}

// calculates the scan of each non-global histogram, saving it in different lines of the vertical scan
__global__ void verticalScanHH(u_int *Hg, u_int *PSv, u_int h, u_int hist_count){
    int posiX = threadIdx.x;         // starts as thread ID

    //--

    while(posiX < h) {
      // Loop while inside the histogram's segments

      int sum = 0;

      for (int posiY = 0; posiY < hist_count; posiY++) {
        PSv[posiX + (posiY*h)] = sum;
        sum += Hg[posiX + (posiY*h)];
      }

      //--

      posiX += blockDim.x;
      // jumps to the next unprocessed column
    }

    //--

    __syncthreads();
}


// calculates the sum of a horizontal vector with a vertical vector
// saves the result inside the matriz
__global__ void partitionKernel(u_int *V, u_int *SHg, u_int *PSv, u_int h, u_int hist_count){
  int posiX = threadIdx.x;         // starts as thread ID

  //--

  while (posiX < h) {
    // Loop while inside the vector horizontal

    // Add value X to the column X
    for (int posiY = 0; posiY < hist_count; posiY++) {
      V[posiX + (posiY*h)] = SHg[posiX] + PSv[posiX + (posiY*h)];
    }

    //--

    posiX += blockDim.x;
    // jumps to the next unprocessed column
  }

  //--

  __syncthreads();
}


//---------------------------------------------------------------------------------


void verifySort(unsigned int *Input, unsigned int *Output, int nElements) {
  thrust::device_vector<unsigned int> d_Input(Input, Input + nElements);
  thrust::device_vector<unsigned int> d_Output(Output, Output + nElements);
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
  //u_int *Input = new unsigned int[nTotalElements];              // Vetor de entrada
  //u_int *Output = new u_int[nTotalElements];                    // Vetor ordenado
  u_int nTotalElements = 18;
  u_int h = 6;
  u_int Input[] = {2, 4, 33, 27, 8, 10, 42, 3, 12, 21, 10, 12, 15, 27, 38, 45, 18, 22};
  u_int *Output = new u_int[nTotalElements];                      // Vetor ordenado
  u_int *stage = new u_int[nTotalElements];                       // Vetor de debug da memoria da gpu
  u_int SEG_SIZE = (ceil((float)nTotalElements/((float)NP*(float)BLOCKS)));

  // Busca menor valor, maior valor e o comprimento do bin
  u_int nMin = *std::min_element(Input, Input + nTotalElements);
  u_int nMax = *std::max_element(Input, Input + nTotalElements);
  u_int binWidth = getBinSize(nMin, nMax, h);

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

  //for (int i = 0; i < nR; ++i) {
    // TODO: TIME STAMP
    blockAndGlobalHisto<<<NP*BLOCKS, THREADS, SEG_SIZE>>>(HH, Hg, h, d_Input, nTotalElements, nMin, nMax, SEG_SIZE, binWidth);
    globalHistoScan<<<1, THREADS>>>(Hg, SHg, h);
    verticalScanHH<<<1, THREADS>>>(HH, PSv, h, NP*BLOCKS);
    partitionKernel<<<1, THREADS>>>(V, SHg, PSv, h, NP*BLOCKS);
    // BitonicSort????
  //}

  // ---

  cudaMemcpy(stage, HH, NP * BLOCKS * h * sizeof(u_int), cudaMemcpyDeviceToHost);
  std::cout << "HH: " << std::endl;
  for (size_t i=0; i<NP*BLOCKS ;i++){
    for (size_t j=0; j<h ;j++)
      std::cout << stage[i*h + j] << " ";
    std::cout << std::endl;
  }

  cudaMemcpy(stage, Hg, h * sizeof(u_int), cudaMemcpyDeviceToHost);
  std::cout << "Hg: " << std::endl;
  for (size_t i=0; i<h ;i++)
    std::cout << stage[i] << " ";
  std::cout << std::endl;

  cudaMemcpy(stage, SHg, h * sizeof(u_int), cudaMemcpyDeviceToHost);
  std::cout << "SHg: " << std::endl;
  for (size_t i=0; i<h ;i++)
    std::cout << stage[i] << " ";
  std::cout << std::endl;

  cudaMemcpy(stage, PSv, NP * BLOCKS * h * sizeof(u_int), cudaMemcpyDeviceToHost);
  std::cout << "PSv: " << std::endl;
  for (size_t i=0; i<NP*BLOCKS ;i++){
    for (size_t j=0; j<h ;j++)
      std::cout << stage[i*h + j] << " ";
    std::cout << std::endl;
  }

  cudaMemcpy(stage, V, NP * BLOCKS * h * sizeof(u_int), cudaMemcpyDeviceToHost);
  std::cout << "V: " << std::endl;
  for (size_t i=0; i<NP*BLOCKS ;i++){
    for (size_t j=0; j<h ;j++)
      std::cout << stage[i*h + j] << " ";
    std::cout << std::endl;
  }

  // ---

  //cudaMemcpy(Output, d_Output, nTotalElements * sizeof(u_int), cudaMemcpyDeviceToHost);
  //verifySort(Input, Output, nTotalElements);

  cudaFree(d_Input);
  cudaFree(d_Output);
  cudaFree(HH);
  cudaFree(Hg);
  cudaFree(SHg);
  cudaFree(PSv);
  cudaFree(V);

  //delete[] Input;
  //delete[] Output;

  return EXIT_SUCCESS;
}

