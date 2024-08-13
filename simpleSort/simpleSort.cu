#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include <cooperative_groups.h>

#include "histogramM.cuh"

namespace cg = cooperative_groups;

#include "chrono.c"

typedef unsigned int u_int;


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

void cudaResetVariables(u_int *HH, u_int *Hg, u_int *SHg, u_int *PSv, u_int h){
  cudaMemset(HH,  0, NB * h * sizeof(u_int));
  cudaMemset(PSv, 0, NB * h * sizeof(u_int));
  cudaMemset(Hg,  0, h * sizeof(u_int));
  cudaMemset(SHg, 0, h * sizeof(u_int));
}


//---------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  if (check_parameters(argc)) { return EXIT_FAILURE; }
  std::srand(std::time(nullptr));

  u_int nTotalElements = std::stoi(argv[1]);                    // Numero de elementos
  u_int h = std::stoi(argv[2]);                                 // Numero de histogramas
  u_int nR = std::stoi(argv[3]);                                // Numero de chamadas do kernel
  u_int *Input = genRandomArray(nTotalElements);                // Vetor de entrada
  //u_int nTotalElements = 18;
  //u_int h = 6;
  //u_int nR = 4;
  //u_int Input[] = {2, 4, 33, 27, 8, 10, 42, 3, 12, 21, 10, 12, 15, 27, 38, 45, 18, 22};
  u_int *Output = new u_int[nTotalElements];                      // Vetor ordenado
  u_int *h_SHg = new u_int[h];                      // Vetor ordenado
  u_int SEG_SIZE = (ceil((float)nTotalElements/((float)NB)));
  chronometer_t chrono_Thrust, chrono_Hist;

  // Busca menor valor, maior valor e o comprimento do bin
  u_int nMin = *std::min_element(Input, Input + nTotalElements);
  u_int nMax = *std::max_element(Input, Input + nTotalElements);
  u_int binWidth = H_getBinSize(nMin, nMax, h);

  // Information printing, a pedido do Zola
  std::cout << "Min: " << nMin << " | Max: " << nMax << std::endl;
  std::cout << "Largura da Faixa: " << binWidth << std::endl;

  // Aloca cores e copia para GPU
  u_int *d_Input, *d_Output, *HH, *Hg, *SHg, *PSv;
  cudaMalloc((void**)&d_Input,  nTotalElements * sizeof(u_int));  // device input data
  cudaMalloc((void**)&d_Output, nTotalElements * sizeof(u_int));  // device input sorted data
  cudaMalloc((void**)&HH,       NB * h         * sizeof(u_int));  // device histogram matrix
  cudaMalloc((void**)&Hg,       h              * sizeof(u_int));  // device histogram sum
  cudaMalloc((void**)&SHg,      h              * sizeof(u_int));  // device histogram prefix sum
  cudaMalloc((void**)&PSv,      NB * h         * sizeof(u_int));  // device matrix vertical prefix sum
  cudaMemcpy(d_Input, Input, nTotalElements * sizeof(u_int), cudaMemcpyHostToDevice);

  chrono_reset(&chrono_Thrust);
  chrono_reset(&chrono_Hist);

  for (int k = 0; k < nR; k++) {
    cudaResetVariables(HH, Hg, SHg, PSv, h);
    chrono_start(&chrono_Hist);
    H_blockAndGlobalHisto<<<NB, THREADS, h*sizeof(u_int)>>>(HH, Hg, h, d_Input, nTotalElements, nMin, nMax, SEG_SIZE, binWidth);
    H_globalHistoScan    <<<1,  THREADS, h*sizeof(u_int)>>>(Hg, SHg, h);
    H_verticalScanHH     <<<NB, THREADS, h*sizeof(u_int)>>>(HH, PSv, h);
    H_PartitionKernel    <<<NB, THREADS, h*sizeof(u_int)>>>(HH, SHg, PSv, h, d_Input, d_Output, nTotalElements, nMin, nMax, SEG_SIZE, binWidth);
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
    chrono_stop(&chrono_Hist);

    verifySort(Input, Output, nTotalElements, &chrono_Thrust, k);
  }

  // ---

  printf("\n----THRUST\n");
  printf("Delta time: " );
  chrono_report_TimeInLoop( &chrono_Thrust, (char *)"thrust sort", nR);

  double thrust_time_seconds = (double) chrono_gettotal( &chrono_Thrust )/((double)1000*1000*1000);
  printf( "Tempo em segundos: %lf s\n", thrust_time_seconds );
  printf( "Vazão: %lf INT/s\n", (nTotalElements)/thrust_time_seconds );
  
  //--

  printf("\n----HISTOGRAM\n");
  printf("Delta time: " );
  chrono_report_TimeInLoop( &chrono_Hist, (char *)"histogram sort", nR);

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

  //delete[] Input;
  delete[] Output;

  return EXIT_SUCCESS;
}

