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

#define NP 28            // Number of processors
#define BLOCKS 2         // Number of blocks per processor
#define THREADS 1024     // Number of threads per block

__global__ void blocksHistoAndScan(unsigned int *HH, unsigned int *PS, int h, unsigned int *Input, int nElements, unsigned int nMin, unsigned int nMax) {
  extern __shared__ unsigned int sharedMemory[];
  unsigned int *sharedHisto = sharedMemory;
  //unsigned int *sharedScan = sharedMemory + h;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // Vetor de histograma
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    sharedHisto[i] = 1;
  } __syncthreads();

  //for (int i = tid; i < nElements; i += stride) {
  //  int bin = (Input[i] - nMin) / (((nMax - nMin)/h) + 1);
  //  atomicAdd(&sharedHisto[bin], 1);
  //} __syncthreads();

  // Adiciona vetor a matrix de histogramas
  if (threadIdx.x == 0) {
    for (int i = 0; i < h; ++i) {
      HH[blockIdx.x * h + i] = sharedHisto[i];
    }
  } __syncthreads();

  //if (threadIdx.x == 0) {
  //  sharedScan[0] = 0;
  //  for (int i = 1; i < h; ++i) {
  //    sharedScan[i] = sharedScan[i - 1] + sharedHisto[i - 1];
  //  }
  //}
  //__syncthreads();

  //if (threadIdx.x == 0) {
  //  for (int i = 0; i < h; ++i) {
  //    PS[blockIdx.x * h + i] = sharedScan[i];
  //  }
  //}
}


//__global__ void globalHistoAndScan(unsigned int *HH, unsigned int *H, unsigned int *PS, unsigned int *P, int h, unsigned int *Input, int nElements, unsigned int nMin, unsigned int nMax) {
//  extern __shared__ unsigned int sharedMemory[];
//  unsigned int *sharedHisto = sharedMemory;
//  unsigned int *sharedScan = sharedMemory + h;
//
//  int tid = threadIdx.x + blockIdx.x * blockDim.x;
//  int stride = blockDim.x * gridDim.x;
//
//  for (int i = threadIdx.x; i < h; i += blockDim.x) {
//    sharedHisto[i] = 0;
//  }
//  __syncthreads();
//
//  for (int i = tid; i < nElements; i += stride) {
//    int bin = (Input[i] - nMin) * h / (nMax - nMin + 1);
//    atomicAdd(&sharedHisto[bin], 1);
//  }
//  __syncthreads();
//
//  if (threadIdx.x == 0) {
//    for (int i = 0; i < h; ++i) {
//      atomicAdd(&H[i], sharedHisto[i]);
//    }
//  }
//  __syncthreads();
//
//  if (threadIdx.x == 0) {
//    sharedScan[0] = 0;
//    for (int i = 1; i < h; ++i) {
//      sharedScan[i] = sharedScan[i - 1] + sharedHisto[i - 1];
//    }
//  }
//  __syncthreads();
//
//  if (threadIdx.x == 0) {
//    for (int i = 0; i < h; ++i) {
//      atomicAdd(&P[i], sharedScan[i]);
//    }
//  }
//}


//__global__ void Partition_kernel(unsigned int *HH, unsigned int *H, unsigned int *PS, unsigned int *P, int h, unsigned int *Output, unsigned int *Input, int nElements, unsigned int nMin, unsigned int nMax) {
//  extern __shared__ unsigned int sharedMemory[];
//  unsigned int *sharedHisto = sharedMemory;
//  unsigned int *sharedScan = sharedMemory + h;
//
//  int tid = threadIdx.x + blockIdx.x * blockDim.x;
//  int stride = blockDim.x * gridDim.x;
//
//  for (int i = threadIdx.x; i < h; i += blockDim.x) {
//    sharedHisto[i] = 0;
//  }
//  __syncthreads();
//
//  for (int i = tid; i < nElements; i += stride) {
//    int bin = (Input[i] - nMin) * h / (nMax - nMin + 1);
//    int pos = P[bin] + atomicAdd(&sharedHisto[bin], 1);
//    Output[pos] = Input[i];
//  }
//}


void verifySort(unsigned int *Input, unsigned int *Output, int nElements) {
  thrust::device_vector<unsigned int> d_Input(Input, Input + nElements);
  thrust::device_vector<unsigned int> d_Output(Output, Output + nElements);
  thrust::sort(d_Input.begin(), d_Input.end());

  bool isSorted = thrust::equal(d_Input.begin(), d_Input.end(), d_Output.begin());

  if (isSorted) { std::cout << "Sort verification: SUCCESS" << std::endl; } 
  else          { std::cout << "Sort verification: FAILURE" << std::endl; }
}


int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: ./simpleSort <nTotalElements> <h> <nR>" << std::endl;
    return EXIT_FAILURE;
  }

  std::srand(std::time(nullptr));
  //int nTotalElements = std::stoi(argv[1]);                    // Numero de elementos
  //int h = std::stoi(argv[2]);                                 // Numero de histogramas
  int nTotalElements = 18;
  int h = 6;
  int nR = std::stoi(argv[3]);                                // Numero de chamadas do kernel
  //unsigned int *Input = new unsigned int[nTotalElements];     // Vetor de entrada
  unsigned int Input[] = {2, 4, 33, 27, 8, 10, 42, 3, 12, 21, 10, 12, 15, 27, 38, 45, 18, 22};
  unsigned int *Output = new unsigned int[nTotalElements];    // Vetor ordenado
  unsigned int *stage = new unsigned int[nTotalElements];     // Vetor de debug da memoria da gpu

  //// Preenche vetor
  //for (int i = 0; i < nTotalElements; ++i) {
  //  int a = std::rand() % 50;
  //  int b = std::rand();
  //  unsigned int v = a * 100 + b;
  //  Input[i] = v;
  //}

  // Busca menor valor e maior valor com thrust
  unsigned int nMin = *std::min_element(Input, Input + nTotalElements);
  unsigned int nMax = *std::max_element(Input, Input + nTotalElements);

  // Alocacores da GPU
  unsigned int *d_Input, *d_Output, *HH, *PS, *H, *P;
  cudaMalloc((void**)&d_Input,  nTotalElements * sizeof(unsigned int));
  cudaMalloc((void**)&d_Output, nTotalElements * sizeof(unsigned int));
  cudaMalloc((void**)&HH,       nTotalElements * sizeof(unsigned int)); // assuming NP=1 for nb=NP*2
  cudaMalloc((void**)&PS,       nTotalElements * sizeof(unsigned int));
  cudaMalloc((void**)&H,        h * sizeof(unsigned int));
  cudaMalloc((void**)&P,        h * sizeof(unsigned int));

  // Copia para memoria global
  cudaMemcpy(d_Input, Input, nTotalElements * sizeof(unsigned int), cudaMemcpyHostToDevice);

  std::cout << "Vetor: ";
  for (size_t i=0; i<nTotalElements ;i++)
    std::cout << Input[i] << " ";
  std::cout << std::endl;
  std::cout << "nMin: " << nMin << std::endl;
  std::cout << "nMax: " << nMax << std::endl;

  //for (int i = 0; i < nR; ++i) {
    blocksHistoAndScan<<<NP*BLOCKS, THREADS>>>(HH, PS, h, d_Input, nTotalElements, nMin, nMax);
    //globalHistoAndScan<<<NP*BLOCKS, THREADS>>>(HH, H, PS, P, h, d_Input, nTotalElements, nMin, nMax);
    //Partition_kernel<<<NP*BLOCKS, THREADS>>>(HH, H, PS, P, h, d_Output, d_Input, nTotalElements, nMin, nMax);
  //}

  cudaMemcpy(stage, HH, nTotalElements * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  std::cout << "HH: ";
  for (size_t i=0; i<nTotalElements ;i++)
    std::cout << stage[i] << " ";
  std::cout << std::endl;

  //cudaMemcpy(Output, d_Output, nTotalElements * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  //verifySort(Input, Output, nTotalElements);

  cudaFree(d_Input);
  cudaFree(d_Output);
  cudaFree(HH);
  cudaFree(PS);
  cudaFree(H);
  cudaFree(P);

  //delete[] Input;
  //delete[] Output;

  return EXIT_SUCCESS;
}

