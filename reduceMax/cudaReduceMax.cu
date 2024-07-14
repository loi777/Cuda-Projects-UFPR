#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "chrono.c"
#include "log.c"

#include "cuPrintf.cuh"
#include "cuPrintf.cu"


#define NP 28            // Number of processors
#define BLOCKS 2         // Number of blocks per processor
#define THREADS 1024     // Number of threads per block
#define CHECK(A, M, ...) \
  check((A), __FILE__, __LINE__, __func__, (M), ##__VA_ARGS__)

typedef unsigned int u_int;

#define MAX(a,b) ((a)>(b) ? (a) : (b))

//=================================================================================



// funcao atomica para Max de float retirada diretamente da internet
__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    return __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
}



//-----------------------------------------------------------


__global__ void reduceMax_persist(float *d_max, float *input, int nElements) {

  int vectorSize = (blockDim.x * 2);                      // every block loop is going to MAX this range
  int totalBlockSize = vectorSize * gridDim.x;            // the total range that all blocks use together

  int idx = (vectorSize * blockIdx.x) + (threadIdx.x * 2);  // separate the threads with a position vacant between

  // Start processing blocks in the vector.

  while (idx < nElements) {                                 // PERSISTENT BLOCK LOOP

    for(int stride=1; stride <= vectorSize; stride *= 2) {  // REDUCE LOOP
        if (idx % (2*stride) == 0) {            // initially all of them MAX the next index
          if ((idx + stride) < nElements) {
            continue;                           // avoids access violation
          }

          input[idx] = MAX(input[idx], input[idx + stride]);
        }
        
        __syncthreads();                        // must sync after each reduce loop
    }

    //-------------------------
    // Save max of this block with atomic

    if (threadIdx.x == 0) {
      *d_max = atomicMaxFloat(d_max, input[idx]);
    }

    __syncthreads();

    //-------------------------

    idx += totalBlockSize;      // go to the next unprocessed block
  }
}



__global__ void reduceMax_persist_Atomic(float *max, float *input, int nElements) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < nElements) {
      for(int stride=1; stride < nElements; stride *= 2) {
          if (idx % (2*stride) == 0) {
              input[idx] = atomicMaxFloat(&input[idx], input[idx + stride]);
          }
          __syncthreads();
      }
  }

  // Final comparison utilizing ATOMIC operations
  // We compare the "winner" of every block against the other
  
  *max = atomicMaxFloat(max, input[0]);        // after making index 0 the max value, transfer to max
}



//-----------------------------------------------------------



inline void generateRandArray(u_int numElements, float* h_input, float* max) {
  // Initialize the host input vectors
  int a;
  int b;

  for (int i = 0; i < numElements; ++i) {
    a = rand();
    b = rand();

    h_input[i] = a * 100.0 + b;

    if (h_input[i] > *max) {
      *max = h_input[i];
    }

    printf("%f\n", h_input[i]);

  }
}



//-----------------------------------------------------------



__host__ __forceinline__ void checkProcessFailure() {
  cudaError_t err = cudaSuccess;            // Check return values for CUDA calls

  err = cudaGetLastError();
  if ( CHECK(err != cudaSuccess, "Failed to launch reduceMax kernel (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);
}



__host__ __forceinline__ void checkResultFailure(float max, float h_max) {
  if ( max != h_max ) {
    fprintf(stderr, "Result verification failed!\n");
    fprintf(stderr, "Max should be: %f\nBut is: %f\n", max, h_max);
    exit(EXIT_FAILURE);
  }
}



__host__ __forceinline__ void getDeviceMax(float* h_max, float* d_max) {
  cudaError_t err = cudaSuccess;            // Check return values for CUDA calls

  err = cudaMemcpy(h_max, d_max, sizeof(u_int), cudaMemcpyDeviceToHost);
  if ( CHECK(err != cudaSuccess, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);
}



__host__ __forceinline__ void copyHostToDeviceVector(float* d_input, float* h_input, size_t size) {
  cudaError_t err = cudaSuccess;            // Check return values for CUDA calls

  err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
  if ( CHECK(err != cudaSuccess, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);
}



__host__ __forceinline__ void resetDeviceMax(float* d_input, float* h_input) {
  cudaError_t err = cudaSuccess;            // Check return values for CUDA calls

  err = cudaMemcpy(d_input, h_input, 1, cudaMemcpyHostToDevice);
  if ( CHECK(err != cudaSuccess, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);
}



//-----------------------------------------------------------



inline void getInput(int argc, char **argv, u_int* numElements, u_int* nR) {
  if (argc >= 2) {
    *numElements = atoi(argv[1]);
  } else {
    printf("AVISO: sem parametro de tamanho, default: 1.000\n\n");
    *numElements = 1000;
  }

  if (argc >= 3) {
    *nR = atoi(argv[2]);
  } else {
    printf("AVISO: sem parametro de repeticao, default: 30\n\n");
    *nR = 30;
  }
}



//=================================================================================



int main(int argc, char **argv) {
  cudaError_t err = cudaSuccess;            // Check return values for CUDA calls
  float *h_input = NULL, *d_input = NULL;   // Host and device vectors
  float h_max, *d_max;                      // Host and device max
  float max = 0;                            // max value

  u_int numElements, nR;
  getInput(argc, argv, &numElements, &nR);  // get inputs

  chronometer_t chrono_Normal;                     // Chronometer
  chronometer_t chrono_Atomic;                     // Chronometer
  chronometer_t chrono_Thrust;                     // Chronometer

  printf("Running reduceMax for %d elements\n", numElements);
  size_t size = numElements * sizeof(float);


  //------------------------ INICIA VARIAVEIS LOCAIS


  // Allocate the host input vector A and check
  h_input = (float *)malloc(size);
  if ( CHECK(h_input == NULL, "Failed to allocate host vectors!\n") )
    exit(EXIT_FAILURE);

  // Initialize the host input vectors
  srand(time(NULL));                            // properly randomise the code
  generateRandArray(numElements, h_input, &max);

  // Restart chrono timers
  chrono_reset(&chrono_Normal);
  chrono_reset(&chrono_Atomic);
  chrono_reset(&chrono_Thrust);


  //------------------------ COPIA DADOS PRA GPU


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
  copyHostToDeviceVector(d_input, h_input, size);

  // Initialize thrust variables
  thrust::device_ptr<float> thrust_d_ptr(d_input);
  thrust::device_vector<float> thrust_d_input(thrust_d_ptr, thrust_d_ptr+numElements);

  printf("Launching CUDA kernels with %d blocks of %d threads\n", NP*BLOCKS, THREADS);


  // EXECUTE PERSIST ============================


  printf("\n === EXECUTANDO KERNEL PERSIST ===\n");
  
  for (int i = 0; i < nR; ++i) {

    printf("Loop: %d \n\n", i);

    //

    h_max = 0;
    resetDeviceMax(d_max, &h_max);

    //-----

    chrono_start(&chrono_Normal);

    reduceMax_persist<<<NP*BLOCKS, THREADS>>>(d_max, d_input, numElements);

    cudaDeviceSynchronize();
    chrono_stop(&chrono_Normal);

    //-----

    copyHostToDeviceVector(d_input, h_input, numElements);

    checkProcessFailure();

    // Copy device max to host max
    getDeviceMax(&h_max, d_max);

    // Verify that the result is correct
    checkResultFailure(max, h_max);
  }


  // EXECUTE ATOMIC =============================


  //printf("\n === EXECUTANDO KERNEL ATOMIC ===\n");
  //
  //for (int i = 0; i < nR; ++i) {

    //h_max = 0;
    //resetDeviceMax(d_max, &h_max);

    //-----
  //  chrono_start(&chrono_Atomic);
//
  //  reduceMax_persist_Atomic<<<NP*BLOCKS, THREADS>>>(d_max, d_input, numElements);
//
  //  cudaDeviceSynchronize();
  //  chrono_stop(&chrono_Atomic);
//
  //  //-----
//
  //  copyHostToDeviceVector(d_input, h_input, numElements);
//
  //  checkProcessFailure();
//
  //  // Copy device max to host max
  //  getDeviceMax(&h_max, d_max);
//
  //  // Verify that the result is correct
  //  checkResultFailure(max, h_max);
  //}


  // EXECUTE THRUST =============================


  printf("\n === EXECUTANDO KERNEL THRUST ===\n");

  for (int i = 0; i < nR; ++i) {
    chrono_start( &chrono_Thrust );

    //h_max = *(thrust::max_element(thrust_d_input.begin(), thrust_d_input.end()));
    h_max = thrust::reduce(thrust_d_input.begin(), thrust_d_input.end(), -INFINITY, thrust::maximum<float>());

    cudaDeviceSynchronize();
    chrono_stop( &chrono_Thrust );

    //-----

    // Verify that the result is correct
    checkResultFailure(max, h_max);
  }


  // IMPRIME RESULTADOS ===================================


  printf("\n === RESULTADOS ===\n");

  //--

  printf("\n----THRUST\n");
  printf("Delta time: " );
  chrono_report_TimeInLoop( &chrono_Thrust, (char *)"thrust max_element", nR);

  double thrust_time_seconds = (double) chrono_gettotal( &chrono_Thrust )/((double)1000*1000*1000);
  printf( "Tempo em segundos: %lf s\n", thrust_time_seconds );
  printf( "Vazão: %lf INT/s\n", (numElements)/thrust_time_seconds );

  //--

  printf("\n----PERSIST\n");
  printf("Delta time: " );
  chrono_report_TimeInLoop( &chrono_Normal, (char *)"reduceMax_persist", nR);

  double reduce_time_seconds = (double) chrono_gettotal( &chrono_Normal )/((double)1000*1000*1000);
  printf( "Tempo em segundos: %lf s\n", reduce_time_seconds );
  printf( "Vazão: %lf INT/s\n", (numElements)/reduce_time_seconds );

  printf("\n--Tempo em relacao ao Thrust\n");
  printf("Em segundos: %lf\n", reduce_time_seconds - thrust_time_seconds);
  printf("Em porcento: %d\n", (int)((thrust_time_seconds/reduce_time_seconds)*100.0));

  //--

  printf("\n----ATOMIC\n");
  printf("Delta time: " );
  chrono_report_TimeInLoop( &chrono_Atomic, (char *)"reduceMax_atomic_persist", nR);

  double atomic_time_seconds = (double) chrono_gettotal( &chrono_Atomic )/((double)1000*1000*1000);
  printf( "Tempo em segundos: %lf s\n", atomic_time_seconds );
  printf( "Vazão: %lf INT/s\n", (numElements)/atomic_time_seconds );

  printf("\n--Tempo em relacao ao Thrust\n");
  printf("Em segundos: %lf\n", atomic_time_seconds - thrust_time_seconds);
  printf("Em porcento: %d\n", (int)((thrust_time_seconds/atomic_time_seconds)*100.0));
  
  
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
