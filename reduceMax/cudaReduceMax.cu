#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "chrono.c"
#include "log.c"

#define NP 28            // Number of processors
#define BLOCKS 2         // Number of blocks per processor
#define THREADS 1024     // Number of threads per block
#define CHECK(A, M, ...) \
  check((A), __FILE__, __LINE__, __func__, (M), ##__VA_ARGS__)

typedef unsigned int u_int;

__global__ void reduceMax_persist(float *max, float *input, int nElements) {
  u_int i;
  #define INITIAL (blockDim.x * blockIdx.x + threadIdx.x)
  #define NTA (gridDim.x * blockDim.x)

  for (i=INITIAL; i<nElements ;i+=NTA)
    if (input[i] > *max)
      *max = input[i];
}


__global__ void reduceMax_atomic_persist(float *max, float *input, int nElements) {
  u_int i;
  #define INITIAL (blockDim.x * blockIdx.x + threadIdx.x)
  #define NTA (gridDim.x * blockDim.x)

  for (i=INITIAL; i<nElements ;i+=NTA)
    if (input[i] > *max)
      *max = input[i];
}


int main(int argc, char **argv) {
  cudaError_t err = cudaSuccess;            // Check return values for CUDA calls
  float *h_input = NULL, *d_input = NULL;   // Host and device vectors
  float h_max, *d_max;                      // Host and device max
  float max = 0;                            // max value
  u_int numElements = atoi(argv[1]);
  u_int nR = atoi(argv[2]);
  chronometer_t chrono;                     // Chronometer

  printf("Running reduceMax for %d elements\n", numElements);
  size_t size = numElements * sizeof(float);

  // INICIA VARIAVEIS LOCAIS
  
  // Allocate the host input vector A and check
  h_input = (float *)malloc(size);
  if ( CHECK(h_input == NULL, "Failed to allocate host vectors!\n") )
    exit(EXIT_FAILURE);

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_input[i] = rand() % 100;
    if ( h_input[i] > max ) { max = h_input[i]; }
  }

  // COPIA DADOS PRA GPU

  // Allocate the device input vector A
  d_input = NULL;
  err = cudaMalloc((void **)&d_input, size);
  if ( CHECK(err != cudaSuccess, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);
  d_max = NULL;
  err = cudaMalloc((void **)&d_max, sizeof(u_int));
  if ( CHECK(err != cudaSuccess, "Failed to allocate device max (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);

  // Copy the host input vectors A and B in host memory to the device input vectors in device memory
  printf("Copying data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
  if ( CHECK(err != cudaSuccess, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);

  // Initialize thrust variables
  thrust::device_ptr<float> thrust_d_ptr(d_input);
  thrust::device_vector<float> thrust_d_input(thrust_d_ptr, thrust_d_ptr+numElements);

  printf("Launching CUDA kernels with %d blocks of %d threads\n", NP*BLOCKS, THREADS);

  // EXECUTE PERSIST ============================

  printf("\n === EXECUTANDO KERNEL PERSIST ===\n");
  chrono_reset(&chrono);
  chrono_start(&chrono);
  for (int i = 0; i < nR; ++i) 
    reduceMax_persist<<<NP*THREADS, THREADS>>>(d_max, d_input, numElements);
  cudaDeviceSynchronize();
  chrono_stop(&chrono);
  err = cudaGetLastError();
  if ( CHECK(err != cudaSuccess, "Failed to launch reduceMax_persist kernel (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);

  // Copy device max to host max
  err = cudaMemcpy(&h_max, d_max, sizeof(u_int), cudaMemcpyDeviceToHost);
  if ( CHECK(err != cudaSuccess, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);

  // Verify that the result is correct
  if ( max != h_max ) {
    fprintf(stderr, "Result verification failed!\n");
    exit(EXIT_FAILURE);
  } else { printf("Max value: %.6f\n", h_max); }

  printf("Tempo médio por ativação do kernel" );
  chrono_report_TimeInLoop( &chrono, (char *)"reduceMax_persist", nR);
  double reduce_time_seconds = (double) chrono_gettotal( &chrono )/((double)1000*1000*1000);
  printf( "Total_time_in_seconds: %lf s\n", reduce_time_seconds );
  printf( "Throughput: %lf INT/s\n", (numElements)/reduce_time_seconds );

  // EXECUTE ATOMIC =============================

  printf("\n === EXECUTANDO KERNEL ATOMIC ===\n");
  chrono_reset(&chrono);
  chrono_start(&chrono);
  for (int i = 0; i < nR; ++i) 
    reduceMax_persist<<<NP*THREADS, THREADS>>>(d_max, d_input, numElements);
  cudaDeviceSynchronize();
  chrono_stop(&chrono);
  err = cudaGetLastError();
  if ( CHECK(err != cudaSuccess, "Failed to launch reduceMax_persist kernel (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);

  // Copy device max to host max
  err = cudaMemcpy(&h_max, d_max, sizeof(u_int), cudaMemcpyDeviceToHost);
  if ( CHECK(err != cudaSuccess, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);

  // Verify that the result is correct
  if ( max != h_max ) {
    fprintf(stderr, "Result verification failed!\n");
    exit(EXIT_FAILURE);
  } else { printf("Max value: %.6f\n", h_max); }

  printf("Tempo médio por ativação do kernel" );
  chrono_report_TimeInLoop( &chrono, (char *)"reduceMax_atomic_persist", nR);

  double atomic_time_seconds = (double) chrono_gettotal( &chrono )/((double)1000*1000*1000);
  printf( "Total_time_in_seconds: %lf s\n", atomic_time_seconds );
  printf( "Throughput: %lf INT/s\n", (numElements)/atomic_time_seconds );

  // EXECUTE THRUST =============================

  printf("\n === EXECUTANDO KERNEL THRUST ===\n");
  chrono_reset(&chrono);
  chrono_start( &chrono );
  for (int i = 0; i < nR; ++i)
    h_max = *(thrust::max_element(thrust_d_input.begin(), thrust_d_input.end()));
  cudaDeviceSynchronize();
  chrono_stop( &chrono );

  // Verify that the result is correct
  if ( max != h_max ) {
    fprintf(stderr, "Result verification failed!\n");
    exit(EXIT_FAILURE);
  } else { printf("Max value: %.6f\n", h_max); }

  printf("Tempo médio por ativação do kernel" );
  chrono_report_TimeInLoop( &chrono, (char *)"thrust max_element", nR);

  double thrust_time_seconds = (double) chrono_gettotal( &chrono )/((double)1000*1000*1000);
  printf( "Total_time_in_seconds: %lf s\n", thrust_time_seconds );
  printf( "Throughput: %lf INT/s\n", (numElements)/thrust_time_seconds );

  // FINALIZA ===================================

  // Free device and host memory
  err = cudaFree(d_input);
  if ( CHECK(err != cudaSuccess, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);
  err = cudaFree(d_max);
  if ( CHECK(err != cudaSuccess, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);
  free(h_input);

  printf("\nTEST FINISHED GRACIOUSLY\n");

  return 0;
}

