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

#define MAX(a,b) ((a)>(b) ? (a) : (b))

//=========================================================================


// funcao atomica para Max de float retirada diretamente da internet
__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}



//---------------------



__global__ void reduceMax_persist(float *max, float *input, int nElements) {
  u_int blockSize = THREADS*2;                      // the size of the vector segment to reduce
  u_int blockStartPosi = blockIdx.x * blockSize;    // the starting index of current block
  u_int startIndexAdd = blockSize * NP*BLOCKS;         // every loop adds to block start position

  u_int threadsActive;                              // how many threads the block is using

  u_int indexToCompare;                             // used to know where to compare
  u_int currentI;                                   // this saves a few calculations each loop

  // Initial loop where we scan MOST of the vector
  // there will be 1 value to compare at the index where every block started

  for (; blockStartPosi < nElements; blockStartPosi += startIndexAdd) {
    // OUTER LOOP, where we increment the position of every block

    indexToCompare = 1;
    currentI = startIndexAdd + (threadIdx.x * 2);

    // INSIDE LOOP, where threads compare values inside block
    for (threadsActive = THREADS; threadsActive > 0; threadsActive /= 2) {
      if (threadsActive < threadIdx.x) {
        continue;     // skip this thread, for it is inactive
      }

      if (currentI > nElements) {
        continue;     // skip, because this thread is outside the current array
      }

      input[currentI] = MAX(input[currentI] , input[currentI + (indexToCompare)]);

      indexToCompare *= 2;
      currentI *= 2;
    }

    // Exiting the inside loop there will be missing 1 last process
    // let thread 0 do this last comparison
    if (threadIdx.x == 0) {
      input[currentI] = MAX(input[currentI] , input[currentI + (indexToCompare)]);
    }
  }

  // Final comparison utilizing ATOMIC operations
  // We compare the "winner" of every block against the other

  if (threadIdx.x == 0) {   // only first thread does this final comparison
    
  }

}



__global__ void reduceMax_atomic_persist(float *max, float *input, int nElements) {
  u_int i;
  #define INITIAL (blockDim.x * blockIdx.x + threadIdx.x)
  #define NTA (gridDim.x * blockDim.x)

  for (i=INITIAL; i<nElements ;i+=NTA)
    if (input[i] > *max)
      *max = input[i];
}



//-------------------



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

  }
}



//-------------------



__host__ __forceinline__ void checkProcessFailure() {
  cudaError_t err = cudaSuccess;            // Check return values for CUDA calls

  err = cudaGetLastError();
  if ( CHECK(err != cudaSuccess, "Failed to launch reduceMax_persist kernel (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);
}



__host__ __forceinline__ void checkResultFailure(float max, float h_max) {
  if ( max != h_max ) {
    fprintf(stderr, "Result verification failed!\n");
    fprintf(stderr, "Max should be: %f\nBut is: %f\n", max, h_max);
    exit(EXIT_FAILURE);
  } else { printf("Max value: %.6f\n", h_max); }
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



//-------------------


__host__ __forceinline__ void getInput(int argc, char **argv, u_int* numElements, u_int* nR) {
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


//=========================================================================



int main(int argc, char **argv) {
  cudaError_t err = cudaSuccess;            // Check return values for CUDA calls
  float *h_input = NULL, *d_input = NULL;   // Host and device vectors
  float h_max, *d_max;                      // Host and device max
  float max = 0;                            // max value

  u_int numElements, nR;
  getInput(argc, argv, &numElements, &nR);  // obtem inputs

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
  generateRandArray(numElements, h_input, &max);

  // Reinicia os chronos
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
  //copyHostToDeviceVector(d_input, h_input, size);
  err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
  if ( CHECK(err != cudaSuccess, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err)) )
    exit(EXIT_FAILURE);

  // Initialize thrust variables
  thrust::device_ptr<float> thrust_d_ptr(d_input);
  thrust::device_vector<float> thrust_d_input(thrust_d_ptr, thrust_d_ptr+numElements);

  printf("Launching CUDA kernels with %d blocks of %d threads\n", NP*BLOCKS, THREADS);

  // EXECUTE PERSIST ============================

  //printf("\n === EXECUTANDO KERNEL PERSIST ===\n");
  //
  //for (int i = 0; i < nR; ++i) {
  //  chrono_start(&chrono_Normal);
  //
  //  reduceMax_persist<<<NP*THREADS, THREADS>>>(d_max, d_input, numElements);
  //
  //  cudaDeviceSynchronize();
  //  chrono_stop(&chrono_Normal);
  //
  //  copyHostToDeviceVector(d_input, h_input, size);   // reinicia o vetor que foi alterado
  //}
  //
  //// check for error
  //checkProcessFailure();
  //
  //// Copy device max to host max
  //getDeviceMax(&h_max, d_max);
  //
  //// Verify that the result is correct
  //checkResultFailure(max, h_max);


  // EXECUTE ATOMIC =============================

  printf("\n === EXECUTANDO KERNEL ATOMIC ===\n");

  for (int i = 0; i < nR; ++i) {
    chrono_start(&chrono_Atomic);

    reduceMax_persist<<<NP*THREADS, THREADS>>>(d_max, d_input, numElements);

    cudaDeviceSynchronize();
    chrono_stop(&chrono_Atomic);

    //copyHostToDeviceVector(d_input, h_input, size);   // reinicia o vetor que será alterado
  }

  // check for error
  checkProcessFailure();

  // Copy device max to host max
  getDeviceMax(&h_max, d_max);

  // Verify that the result is correct
  checkResultFailure(max, h_max);


  // EXECUTE THRUST =============================


  printf("\n === EXECUTANDO KERNEL THRUST ===\n");

  for (int i = 0; i < nR; ++i) {
    chrono_start( &chrono_Thrust );

    h_max = *(thrust::max_element(thrust_d_input.begin(), thrust_d_input.end()));

    cudaDeviceSynchronize();
    chrono_stop( &chrono_Thrust );
  }

  // Verify that the result is correct
  checkResultFailure(max, h_max);


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

  printf("--Tempo em relacao ao Thrust");
  printf("Em segundos: %lf", reduce_time_seconds - thrust_time_seconds);
  printf("Em porcento: %lf", (thrust_time_seconds/reduce_time_seconds)*100.0);

  //--

  printf("\n----ATOMIC\n");
  printf("Delta time: " );
  chrono_report_TimeInLoop( &chrono_Atomic, (char *)"reduceMax_atomic_persist", nR);

  double atomic_time_seconds = (double) chrono_gettotal( &chrono_Atomic )/((double)1000*1000*1000);
  printf( "Tempo em segundos: %lf s\n", atomic_time_seconds );
  printf( "Vazão: %lf INT/s\n", (numElements)/atomic_time_seconds );

  printf("--Tempo em relacao ao Thrust");
  printf("Em segundos: %lf", atomic_time_seconds - thrust_time_seconds);
  printf("Em porcento: %lf", (thrust_time_seconds/atomic_time_seconds)*100.0);


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

