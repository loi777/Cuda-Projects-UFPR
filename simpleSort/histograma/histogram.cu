#include <iostream>
#include <cuda.h>
#include <limits>

typedef unsigned int u_int;

#define BLOCKS 8                            // one block for one histogram
#define THREADS 5                           // n of threads

#define ARRAYSIZE 10                        // Size of the input array
#define HIST_SEGMENTATIONS 5                // BINS number

#define SEG_SIZE (ceil((float)ARRAYSIZE/(float)BLOCKS))   // Every block will solve this size, minimun of 1

#define HISTOGRAM (BLOCKS*HIST_SEGMENTATIONS)             // the full histogram, block:y | segmentation:x

#define BINSTART(min, binSize, i) ((u_int)((binWidth*i)+min))
#define BINEND(min, binSize, i) ((u_int)(BINSTART(min, binSize, (i+1))-1))
#define BINFIND(min, max, val, binSize, binQtd) (val >= max ? binQtd-1 : (val - min) / binSize)

//---------------------------------------------------------------



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



//---------------------------------------------------------------



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


// returns the min value of an Array
u_int getMin(u_int* Array, int nElem) {
  u_int min = UINT_MAX;

  for (int i = 0; i < nElem; ++i) {
    if (Array[i] < min) {
        min = Array[i];
    }
  }

  return min;
}


// returns the max value of an Array
u_int getMax(u_int* Array, int nElem) {
  u_int max = 0;    // minimun value of an unsigned variable is 0

  for (int i = 0; i < nElem; ++i) {
    if (Array[i] > max) {
        max = Array[i];
    }
  }

  return max;
}



//---------------------------------------------------------------



// FOR INTERNAL AND DEBUGG USE
// Debugg function to print an array
void intPrintArray(int* a, int size) {
    // Print the generated array, do not allow this with arrays of billions
    for (int i = 0; i < size; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}


// FOR INTERNAL AND DEBUGG USE
// Debugg function to print an array
void uintPrintArray(u_int* a, int size) {
    // Print the generated array, do not allow this with arrays of billions
    for (int i = 0; i < size; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}


// FOR INTERNAL AND DEBUGG USE
// Debugg function to print an array
void printSegmentations(u_int min, u_int max, u_int* a, int size, int segCount) {
    u_int binWidth = getBinSize(min, max, segCount);  

    //--

    std::cout << "min: " << min << " | max: " << max << std::endl;
    std::cout << "Bin Size: " << binWidth << std::endl;
    for (int i = 0; i < segCount; i++) {
        std::cout << "Seg|Bin [" << i << "]: " << BINSTART(min, binWidth, i) << " to " << BINEND(min, binWidth, i) << "\n";
    }
    std::cout << std::endl;
}



//---------------------------------------------------------------



// Kernel para calcular histogramas em particoes
// Cada bloco eh responsavel por um histograma (linha da matriz)
__global__ void calculateHistogram(const u_int *input, int *histograms, int *histogram_T, int arraySize, int segSize, int segCount, u_int minVal, u_int maxVal, u_int binWidth) {
    // Alloca shared memory para UM histograma
    extern __shared__ int sharedHist[];

    __syncthreads();  // threads will wait for the shared memory to be finished

    //---

    // Inicio da particao no vetor
    int blcStart = (blockIdx.x * segSize);    // bloco positionado na frente daquele que veio anterior a ele
    int thrPosi = threadIdx.x;              // 1 elemento por thread, starts as exactly the thread.x

    while(thrPosi < segSize && ((blcStart+thrPosi) < arraySize)) {
        // Loop enquanto a thread estiver resolvendo elementos validos dentro do bloco e do array

        u_int val = input[blcStart + thrPosi];    // get value
        int posi = BINFIND(minVal, maxVal, val, binWidth, segCount);
        atomicAdd(&sharedHist[posi], 1);  // add to its corresponding segment
        atomicAdd(&histogram_T[posi], 1);  // add to its corresponding segment

        thrPosi += blockDim.x; // thread pula para frente, garantindo que nao ira processar um valor ja processado
        // saira do bloco quando terminar todos os pixeis dos quais eh responsavel
    }

    __syncthreads();

    //--

    // Passa os resultados da shared memory para matriz
    // deixar isso a cargo da thread 0 eh mais modular que mandar uma pra uma
    if (threadIdx.x == 0) {
      for (int i = 0; i < segCount; i++) {
        atomicAdd(&histograms[(blockIdx.x * segCount) + i], sharedHist[i]); 
      }
    }
    // Y: (blockIdx.x * segCount)
    // X: threadIdx.x

    __syncthreads();
}



//---------------------------------------------------------------



// calculates the scan of the global histogram and saves it into the horizontal scan
__global__ void calculateHorizontalScan(int *histogram_T, int *scan, int segCount) {
    int threadPosi = threadIdx.x;         // starts as thread ID

    //--

    while(threadPosi < segCount) {
      // Loop while inside the histogram

      int sum = 0;

      for (int i = threadPosi-1; i >= 0; i--) {
        // makes the individual sum of every index before this one
        sum += histogram_T[i];
      }

      scan[threadPosi] = sum;

      //--

      threadPosi += blockDim.x; // go to the next element
    }

    //--

    __syncthreads();
}


// calculates the scan of each non-global histogram, saving it in different lines of the vertical scan
__global__ void calculateVerticalScan(int *histograms, int *Vscan, int segCount, int hist_count) {
    int posiX = threadIdx.x;         // starts as thread ID

    //--

    while(posiX < segCount) {
      // Loop while inside the histogram's segments

      int sum = 0;

      for (int posiY = 0; posiY < hist_count; posiY++) {
        Vscan[posiX + (posiY*segCount)] = sum;
        sum += histograms[posiX + (posiY*segCount)];
      }

      //--

      posiX += blockDim.x;
      // jumps to the next unprocessed column
    }

    //--

    __syncthreads();
}



//---------------------------------------------------------------



// calculates the sum of a horizontal vector with a vertical vector
// saves the result inside the matriz
__global__ void calculateVectorSum(int* matriz, int* horVec, int* vertVec, int horSize, int vertSize) {
  int posiX = threadIdx.x;         // starts as thread ID

  //--

  while(posiX < horSize) {
    // Loop while inside the vector horizontal

    // Add value X to the column X
    for (int posiY = 0; posiY < vertSize; posiY++) {
      matriz[posiX + (posiY*horSize)] = horVec[posiX] + vertVec[posiX + (posiY*horSize)];
    }

    //--

    posiX += blockDim.x;
    // jumps to the next unprocessed column
  }

  //--

  __syncthreads();
}



//---------------------------------------------------------------



int main() {
    ////======= INPUT VECTORS
    u_int *d_input;
    u_int* h_input = genRandomArray(ARRAYSIZE);

    // Allocate memory on the device
    cudaMalloc((void**)&d_input, ARRAYSIZE * sizeof(u_int));
    cudaMemcpy(d_input, h_input, ARRAYSIZE * sizeof(u_int), cudaMemcpyHostToDevice);

    u_int min = getMin(h_input, ARRAYSIZE);
    u_int max = getMax(h_input, ARRAYSIZE);
    u_int binWidth = getBinSize(min, max, HIST_SEGMENTATIONS);

    uintPrintArray(h_input, ARRAYSIZE);
    printSegmentations(min, max, h_input, ARRAYSIZE, HIST_SEGMENTATIONS);

    ////======= HISTOGRAM

    int *d_histograms;          // each block has a histogram
    int *d_histogram_total;     // the sum of all histograms together
    int h_histograms[BLOCKS][HIST_SEGMENTATIONS] = {0};
    int h_histogram_total[HIST_SEGMENTATIONS] = {0};

    // Allocate memory on the device
    cudaMalloc((void**)&d_histograms, HISTOGRAM * sizeof(int));
    cudaMemset(d_histograms, 0, HISTOGRAM * sizeof(int));  // Initialize histograms to 0

    cudaMalloc((void**)&d_histogram_total, HIST_SEGMENTATIONS * sizeof(int));
    cudaMemset(d_histograms, 0, HIST_SEGMENTATIONS * sizeof(int));  // Initialize histograms to 0

    ////======= SCAN

    int *d_scan;                                    // the scan of the global histogram
    int *d_verticalScan;                            // the scan of each histogram
    int h_scan[HIST_SEGMENTATIONS] = {0};
    int h_verticalScan[BLOCKS][HIST_SEGMENTATIONS] = {0};

    cudaMalloc((void**)&d_scan, HIST_SEGMENTATIONS * sizeof(int));
    cudaMemset(d_scan, 0, HIST_SEGMENTATIONS * sizeof(int));  // Initialize histograms to 0

    cudaMalloc((void**)&d_verticalScan, HISTOGRAM * sizeof(int));
    cudaMemset(d_verticalScan, 0, HISTOGRAM * sizeof(int));  // Initialize histograms to 0

    ////======= VECTORIAL SUM

    int *d_vecSum;                                    // the conbination of Horizontal and Vertical Scan
    int h_vecSum[BLOCKS][HIST_SEGMENTATIONS] = {0};

    cudaMalloc((void**)&d_vecSum, HISTOGRAM * sizeof(int));
    cudaMemset(d_vecSum, 0, HISTOGRAM * sizeof(int));  // Initialize histograms to 0

    ////=======////======= KERNEL 1 - HIST

    // Launch kernel
    calculateHistogram<<<BLOCKS, THREADS, SEG_SIZE>>>(d_input, d_histograms, d_histogram_total, ARRAYSIZE, SEG_SIZE, HIST_SEGMENTATIONS, min, max, binWidth);

    ////=======////======= KERNEL 2+3 - SCAN

    // Launch kernel horizontal scan
    calculateHorizontalScan<<<1, THREADS>>>(d_histogram_total, d_scan, HIST_SEGMENTATIONS);

    // Launch kernel vertical scan
    calculateVerticalScan<<<1, THREADS>>>(d_histograms, d_verticalScan, HIST_SEGMENTATIONS, BLOCKS);

    ////=======////======= KERNEL 4 - VECTOR SUM

    // Launch kernel for vectorial sum
    calculateVectorSum<<<1, THREADS>>>(d_vecSum, d_scan, d_verticalScan, HIST_SEGMENTATIONS, BLOCKS);

    ////=======////======= COPY BACK

    // Copy result back to host
    cudaMemcpy(h_histograms, d_histograms, HISTOGRAM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_histogram_total, d_histogram_total, HIST_SEGMENTATIONS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scan, d_scan, HIST_SEGMENTATIONS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_verticalScan, d_verticalScan, HISTOGRAM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vecSum, d_vecSum, HISTOGRAM * sizeof(int), cudaMemcpyDeviceToHost);

    ////======= PRINT RESULT

    // Print the histograms
    for (int i = 0; i < BLOCKS; i++) {
        std::cout << "Histogram " << i << ": ";
        for (int j = 0; j < HIST_SEGMENTATIONS; j++)
            std::cout << h_histograms[i][j] << " ";
        std::cout << std::endl;
    }

    std::cout << "Histogram_total: ";
    intPrintArray(h_histogram_total, HIST_SEGMENTATIONS);

    std::cout << "Hist Scan: ";
    intPrintArray(h_scan, HIST_SEGMENTATIONS);

    // Print the histograms
    for (int i = 0; i < BLOCKS; i++) {
        std::cout << "Hist Vertical Scan " << i << ": ";
        for (int j = 0; j < HIST_SEGMENTATIONS; j++)
            std::cout << h_verticalScan[i][j] << " ";
        std::cout << std::endl;
    }

    // Print the histograms
    for (int i = 0; i < BLOCKS; i++) {
        std::cout << "Final vector Sum " << i << ": ";  
        for (int j = 0; j < HIST_SEGMENTATIONS; j++)
            std::cout << h_vecSum[i][j] << " ";
        std::cout << std::endl;
    }

    ////======= FREE MEMORY

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_histograms);
    cudaFree(d_histogram_total);
    cudaFree(d_scan);
    cudaFree(d_verticalScan);
    cudaFree(d_vecSum);

    return 0;
}

