
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdlib>

#include "chrono.c"
#define N_REPETICOES 30
#define V_SIZE 32 << 20

int main() {
  float max = 0;

  printf("\n----- exemplo com thrust::max_element para inteiros\n"
         "        modificado com medicoes de tempo\n"
         "        usando pacote chrono.c e calculando \n"
         "        vazão de numeros inteiros de 32Bits ordenados por segundo\n"
         "------------------------------------------\n\n\n" );

  // generate 32M random numbers on the host
  // transfer data to the device
  thrust::host_vector<int> h_vec(V_SIZE);
  thrust::generate(h_vec.begin(), h_vec.end(), rand);
  thrust::device_vector<int> d_vec = h_vec;

  // cria um chonometro para medir thrust::sort
  chronometer_t chrono_para_Sort;  

  // Executa
  chrono_start( &chrono_para_Sort );
  for (int i = 0; i < N_REPETICOES; ++i)
    max = *(thrust::max_element(d_vec.begin(), d_vec.end()));
  cudaDeviceSynchronize();
  chrono_stop( &chrono_para_Sort );

  printf("\n----- reportando o tempo total para\n"
         "as %d ativações do kernel Sort do thrust -------", N_REPETICOES );
  chrono_reportTime( &chrono_para_Sort, (char *)"thrust::sort kernel" );
  printf("\n\n" );    

  // calcular e imprimir a VAZAO (numero de INT/s)
  double total_time_in_seconds = (double) chrono_gettotal( &chrono_para_Sort )/((double)1000*1000*1000);
  printf( "total_time_in_seconds: %lf s\n", total_time_in_seconds );

  double OPS = (V_SIZE)/total_time_in_seconds;
  printf( "Throughput: %lf INT/s\n", OPS );

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  printf("Valor maximo: %f\n", max);
  return 0;
}
