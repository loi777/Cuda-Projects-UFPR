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



// FOR DEBUGG USE
// funcao para imprimir uma array de uint
void H_printArray(u_int* arr, u_int size) {
  std::cout << "ARRAY PRINTING:  ";
  
  //--

  for (int i = 0; i < size; i++) {
    std::cout << arr[i] << " ";
  }
  
  //--

  std::cout << "\n";
}


// FOR DEBUGG USE
// funcao para imprimir uma array de uint do device
__global__ void D_printArray(u_int* d_arr, u_int size) {
  if (threadIdx.x == 0) {
    printf("ARRAY PRINTING:  ");

    //--

    for (int i = 0; i < size; i++) {
      printf("%d ", d_arr[i]);
    }

    //--

    printf("\n");
  }
}



//---------------------------------------------------------------------------------



// FOR INTERNAL USE
// Obtem 2 vetores ordenados e verifica se estao ordenados tanto entre si quanto entre si mesmos
__global__ void verifySort(u_int* d_thr, u_int* d_bit, u_int size) {
  u_int posi = (blockDim.x*blockIdx.x) + threadIdx.x;

  while(posi < size) {
    if (posi > 0) {
      if (d_bit[posi-1] > d_bit[posi])  printf("QUEBRA DE SORT\nBITONIC: ! [%d]%d > [%d]%d\n", posi-1, d_bit[posi-1], posi, d_bit[posi]);
      if (d_thr[posi-1] > d_thr[posi])  printf("QUEBRA DE SORT\nTHRUST: ! [%d]%d > [%d]%d\n", posi-1, d_thr[posi-1], posi, d_thr[posi]);
    }

    posi += blockDim.x;
  }
}


// FOR INTERNAL USE
// uma funcao que combina a criacao de memoria pro cuda e o verify sort
void verifySortProxy(u_int* h_thr, u_int* h_bit, u_int size) {
  u_int *d_thr, *d_bit;
  cudaMalloc((void**)&d_thr, sizeof(u_int) * size);
  cudaMalloc((void**)&d_bit, sizeof(u_int) * size);

  cudaMemcpy(d_thr, h_thr, size * sizeof(u_int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bit, h_bit, size * sizeof(u_int), cudaMemcpyHostToDevice);

  //--
  verifySort<<<1, THREADS >>>(d_thr, d_bit, size);
  //--

  cudaFree(d_thr);
  cudaFree(d_bit);
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



// CPU level recursion function that uses histogram to constantly reduce the size of the array
// when this goes below the shared memory limit then bitonic sort is used to sort the array.
void recursionBitonic(u_int* d_orig, u_int p_start, u_int p_end, u_int histograms) {
  u_int a_size = (p_end-p_start);                               // obtem o tamanho em elementos dessa particao
  u_int h_min = UINT32_MAX;
  u_int h_max = 0;
  H_getDeviceMinMax(d_orig, p_start, a_size, &h_min, &h_max);
  
  u_int binWidth = H_getBinSize(h_min, h_max, histograms);      // obtem as ranges dos conjuntos numericos/bins
  u_int SEG_SIZE = (ceil((float)a_size/((float)NB)));           // obtem o tamanho em elementos do bloco

  //--

  if (a_size < POW2LIMIT) {    // esse segmento eh pequeno o suficiente, ordena com bitonic

    B_bitonicProxy(d_orig+p_start, d_orig+p_start, a_size);

  } else {                    // esse segmento eh mt grande, particiona com histogramas

    u_int *d_part, *d_HH, *d_Hg, *d_horizontalS, *d_verticalS;
    u_int h_horizontalS[histograms];
    cudaMalloc((void**)&d_part,         a_size                  * sizeof(u_int));  // device input partitionec data
    cudaMalloc((void**)&d_HH,           NB * histograms         * sizeof(u_int));  // device histogram matrix
    cudaMalloc((void**)&d_Hg,           histograms              * sizeof(u_int));  // device histogram sum
    cudaMalloc((void**)&d_horizontalS,  histograms              * sizeof(u_int));  // device histogram prefix sum
    cudaMalloc((void**)&d_verticalS,    NB * histograms         * sizeof(u_int));  // device matrix vertical prefix sum

    ////==== ALOCA MEMORIA CUDA
    
    cudaResetVariables(d_HH, d_Hg, d_horizontalS, d_verticalS, histograms);
    H_getHistogram  <<<NB, THREADS, histograms*sizeof(u_int)>>>(d_HH, d_Hg, histograms, d_orig+p_start, a_size, h_min, h_max, SEG_SIZE, binWidth);
    H_horizontalScan<<<NB,  THREADS, histograms*sizeof(u_int)>>>(d_Hg, d_horizontalS, histograms);
    H_verticalScan  <<<NB, THREADS, histograms*sizeof(u_int)>>>(d_HH, d_verticalS, histograms);
    H_Partitioner   <<<NB, THREADS, histograms*sizeof(u_int)>>>(d_HH, d_horizontalS, d_verticalS, histograms, d_orig+p_start, d_part, a_size, h_min, h_max, SEG_SIZE, binWidth);
    cudaMemcpy(d_orig+p_start, d_part, a_size * sizeof(u_int), cudaMemcpyDeviceToDevice); // salva no original para recursao
    cudaMemcpy(h_horizontalS, d_horizontalS, histograms * sizeof(u_int), cudaMemcpyDeviceToHost); // salva no host o histograma horizontal para saber a posicao das part.

    ////==== PARTICIONA USANDO HISTOGRAMA

    cudaFree(d_part);
    cudaFree(d_HH);
    cudaFree(d_Hg);
    cudaFree(d_horizontalS);
    cudaFree(d_verticalS);

    ////==== LIMPA MEMORIA CUDA

    for (int p_hist = 1; p_hist < histograms; p_hist++) {
      recursionBitonic(d_orig, h_horizontalS[p_hist-1]+p_start, h_horizontalS[p_hist]+p_start, histograms);
    }
    recursionBitonic(d_orig, h_horizontalS[histograms-1]+p_start, p_end, histograms);  // o ultimo ponto quebra a logica do loop e eh feito fora

    ////==== CONTINUA A RECURSAO
  }
}



//---------------------------------------------------------------------------------



int main(int argc, char* argv[]) {
  if (check_parameters(argc)) { return EXIT_FAILURE; }
  std::srand(std::time(nullptr));

  u_int nTotalElements = std::stoi(argv[1]);                    // Numero de elementos
  u_int h = std::stoi(argv[2]);                                 // Numero de histogramas/recursao
  u_int nR = std::stoi(argv[3]);                                // Numero de chamadas do kernel
  u_int *h_Input = genRandomArray(nTotalElements);              // Vetor de entrada
  u_int *h_Output_bi = new u_int[nTotalElements];                  // Vetor final BITONIC
  u_int *h_Output_th = new u_int[nTotalElements];                  // Vetor final THRUST

  ////====  GET GLOBAL VARIABLES

  u_int *d_input;
  cudaMalloc((void**)&d_input, nTotalElements * sizeof(u_int));   // device input data

  ////====  GET CUDA MEMORY

  chronometer_t chrono_Thrust, chrono_Hist;
  chrono_reset(&chrono_Thrust);
  chrono_reset(&chrono_Hist);

  ////====  GET CHRONO VARIABLES

  // Information printing, a pedido do Zola
  // variaveis apenas usadas nesse print
  u_int nMin = *std::min_element(h_Input, h_Input + nTotalElements);  // obtem o min dessa particao
  u_int nMax = *std::max_element(h_Input, h_Input + nTotalElements);  // obtem o max dessa particao
  u_int binWidth = H_getBinSize(nMin, nMax, h);                   // obtem as ranges dos conjuntos numericos/bins

  std::cout << "Min: " << nMin << " | Max: " << nMax << std::endl;
  std::cout << "Largura da Faixa: " << binWidth << std::endl;

  ////====  PRINT DE INFORMAÇÃO
  
  for (int i = 0; i < nR; i++) {
    cudaMemcpy(d_input, h_Input, nTotalElements * sizeof(u_int), cudaMemcpyHostToDevice);     // get input array to device

    chrono_start(&chrono_Hist);
    recursionBitonic(d_input, 0, nTotalElements, h);                                          // Begin bitonic sort
    chrono_stop(&chrono_Hist);

    cudaMemcpy(h_Output_bi, d_input, nTotalElements * sizeof(u_int), cudaMemcpyDeviceToHost); // get output array to host
  }

  ////====  BITONIC RECURSION

  for (int i = 0; i < nR; i++) {
    thrust::device_vector<u_int> d_vec(h_Input, h_Input + nTotalElements);  // get input array to thrust

    chrono_start(&chrono_Thrust);

    thrust::sort(d_vec.begin(), d_vec.end());                               // Begin thrust sort

    chrono_stop(&chrono_Thrust);

    thrust::copy(d_vec.begin(), d_vec.end(), h_Output_th);                  // get output array to host 
  }

  ////====  THRUST SORT

  verifySortProxy(h_Output_th, h_Output_bi, nTotalElements);

  ////====  VERIFICA SORT

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

  ////==== PRINT RESULTADOS

  cudaFree(d_input);

  ////==== FREE MEMORY

  return EXIT_SUCCESS;
}

