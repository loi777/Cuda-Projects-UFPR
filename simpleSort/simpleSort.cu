#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include <cooperative_groups.h>

#include "simpleSort.cuh"
#include "histogramM.cuh"
#include "bitonicM.cuh"

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



// TODO :: APLICAR O P_START NOS ARRAYS PARA MOVIMENTALOS NA POSICAO CERTA
void recursionBitonic(u_int* h_array, u_int p_start, u_int p_end, u_int histograms) {
  u_int a_size = (p_end-p_start);                             // obtem o tamanho em elementos dessa particao
  u_int nMin = *std::min_element(h_array, h_array + a_size);  // obtem o min dessa particao
  u_int nMax = *std::max_element(h_array, h_array + a_size);  // obtem o max dessa particao

  u_int binWidth = H_getBinSize(nMin, nMax, histograms);      // obtem as ranges dos conjuntos numericos/bins
  u_int SEG_SIZE = (ceil((float)a_size/((float)NB)));         // obtem o tamanho em elementos 

  //--

  if ((p_end-p_start) < SHAREDLIMIT) {    // esse segmento eh pequeno o suficiente, ordena com bitonic

    //code

  } else {      // esse segmento eh mt grande, particiona com histogramas

    u_int *d_array, *d_partitioned, *d_HH, *d_Hg, *d_horizontalS, *d_verticalS;
    u_int h_horizontalS[histograms];
    cudaMalloc((void**)&d_array,        a_size                  * sizeof(u_int));  // device input data
    cudaMalloc((void**)&d_partitioned,  a_size                  * sizeof(u_int));  // device output partitionec data
    cudaMalloc((void**)&d_HH,           NB * histograms         * sizeof(u_int));  // device histogram matrix
    cudaMalloc((void**)&d_Hg,           histograms              * sizeof(u_int));  // device histogram sum
    cudaMalloc((void**)&d_horizontalS,  histograms              * sizeof(u_int));  // device histogram prefix sum
    cudaMalloc((void**)&d_verticalS,    NB * histograms         * sizeof(u_int));  // device matrix vertical prefix sum

    cudaMemcpy(d_array, h_array, a_size * sizeof(u_int), cudaMemcpyHostToDevice);  // copia do host o input

    ////==== ALOCA MEMORIA CUDA

    
    cudaResetVariables(d_HH, d_Hg, d_horizontalS, d_verticalS, histograms);
    H_getHistogram        <<<NB, THREADS, histograms*sizeof(u_int)>>>(d_HH, d_Hg, histograms, d_array, a_size, nMin, nMax, SEG_SIZE, binWidth);
    H_horizontalScan      <<<1,  THREADS, histograms*sizeof(u_int)>>>(d_Hg, d_horizontalS, histograms);
    H_verticalScan        <<<NB, THREADS, histograms*sizeof(u_int)>>>(d_HH, d_verticalS, histograms);
    H_Partitioner         <<<NB, THREADS, histograms*sizeof(u_int)>>>(d_HH, d_horizontalS, d_verticalS, histograms, d_array, d_partitioned, a_size, nMin, nMax, SEG_SIZE, binWidth);
    cudaMemcpy(h_array, d_partitioned, a_size * sizeof(u_int), cudaMemcpyDeviceToHost);           // salva no host a particao feita
    cudaMemcpy(h_horizontalS, d_horizontalS, histograms * sizeof(u_int), cudaMemcpyDeviceToHost); // salva no host o histograma horizontal para saber a posicao das part.
 

    ////==== PARTICIONA USANDO HISTOGRAMA

    for (int p_hist = 1; p_hist < histograms; p_hist++) {
      recursionBitonic(h_array, h_horizontalS[p_hist-1], h_horizontalS[p_hist], histograms);
    }
    recursionBitonic(h_array, h_horizontalS[histograms-1], p_end, histograms);  // o ultimo ponto quebra a logica do loop e eh feito fora

    ////==== CONTINUA A RECURSAO

    cudaFree(d_array);
    cudaFree(d_partitioned);
    cudaFree(d_HH);
    cudaFree(d_Hg);
    cudaFree(d_horizontalS);
    cudaFree(d_verticalS);

    ////==== LIMPA MEMORIA CUDA
  }
}



//---------------------------------------------------------------------------------



int main(int argc, char* argv[]) {
  if (check_parameters(argc)) { return EXIT_FAILURE; }
  std::srand(std::time(nullptr));

  u_int nTotalElements = std::stoi(argv[1]);                    // Numero de elementos
  u_int h = std::stoi(argv[2]);                                 // Numero de histogramas/recursao
  u_int nR = std::stoi(argv[3]);                                // Numero de chamadas do kernel
  u_int *Input = genRandomArray(nTotalElements);                // Vetor de entrada
  u_int *Output = new u_int[nTotalElements];                    // Vetor final

  ////====  GET GLOBAL VARIABLES

  chronometer_t chrono_Thrust, chrono_Hist;
  chrono_reset(&chrono_Thrust);
  chrono_reset(&chrono_Hist);

  ////====  GET CHRONO VARIABLES

  // Information printing, a pedido do Zola
  u_int nMin = *std::min_element(Input, Input + nTotalElements);  // obtem o min dessa particao
  u_int nMax = *std::max_element(Input, Input + nTotalElements);  // obtem o max dessa particao
  u_int binWidth = H_getBinSize(nMin, nMax, h);                   // obtem as ranges dos conjuntos numericos/bins

  std::cout << "Min: " << nMin << " | Max: " << nMax << std::endl;
  std::cout << "Largura da Faixa: " << binWidth << std::endl;

  ////====  PRINT DE INFORMAÇÃO

  chrono_start(&chrono_Hist);
  //code
  chrono_stop(&chrono_Hist);

  ////====  BITONIC RECURSION

  chrono_start(&chrono_Thrust);
  //code
  chrono_stop(&chrono_Thrust);

  ////====  THRUST SORT

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

  ////==== PRINT RESULTADOS

  //delete[] Input;
  delete[] Output;

  ////==== FREE MEMORY

  return EXIT_SUCCESS;
}

