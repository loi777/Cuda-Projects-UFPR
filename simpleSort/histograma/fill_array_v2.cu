#include <iostream>
#include <cuda.h>
#include <limits>

#define N 18              // Size of the input array
#define HISTOGRAM_BINS 6  // Number of bins in each histogram
#define NUM_HISTOGRAMS 3  // Number of histograms (rows in the output matrix)

// The input array
const int h_input[N] = {2, 4, 33, 27, 8, 10, 42, 3, 12, 21, 10, 12, 15, 27, 38, 45, 18, 22};

// Kernel para calcular histogramas em particoes
// Cada bloco eh responsavel por um histograma (linha da matriz)
__global__ void calculateHistogram(const int *input, int *histograms, int numElements, int numHistograms, int histogramBins, int minVal, int maxVal){
    // Alloca shared memory para UM histograma, tamanho terceiro parametro de chamada
    extern __shared__ int sharedHist[];
    // Inicio da particao no vetor
    int startIdx = blockIdx.x * histogramBins;

    // Verifica overflow dos blocos e das threads
    if ((blockIdx.x > numHistograms) || ((startIdx + threadIdx.x) > numElements)) { return; }
    else {
        // Inicia shared memory
        if (threadIdx.x < histogramBins) { sharedHist[threadIdx.x] = 0; }
        __syncthreads();  // Wait for all threads to initialize shared memory

        // Calcula intervalo de cada bin
        unsigned int binWidth = ((maxVal - minVal) / histogramBins) + 1;

        int value = input[startIdx + threadIdx.x];
        atomicAdd(&sharedHist[(value - minVal) / binWidth], 1);  // Incrementa bin do histograma na shared memory
        __syncthreads();

        // Passa os resultados da shared memory para matriz
        if (threadIdx.x < histogramBins)
            atomicAdd(&histograms[blockIdx.x * histogramBins + threadIdx.x], sharedHist[threadIdx.x]);
    }
}

int main() {
    // Host histograms matrix
    int h_histograms[NUM_HISTOGRAMS][HISTOGRAM_BINS] = {0};

    // Device arrays
    int *d_input, *d_histograms;

    // Allocate memory on the device
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_histograms, NUM_HISTOGRAMS * HISTOGRAM_BINS * sizeof(int));

    // Copy input array to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histograms, 0, NUM_HISTOGRAMS * HISTOGRAM_BINS * sizeof(int));  // Initialize histograms to 0

    // Launch kernel
    int numBlocks = NUM_HISTOGRAMS;            // Each block corresponds to one histogram
    int threadsPerBlock = N / NUM_HISTOGRAMS;  // Each thread handles one element in the partition
    calculateHistogram<<<numBlocks, threadsPerBlock, HISTOGRAM_BINS>>>(d_input, d_histograms, N, NUM_HISTOGRAMS, HISTOGRAM_BINS, 2, 45);

    // Copy result back to host
    cudaMemcpy(h_histograms, d_histograms, NUM_HISTOGRAMS * HISTOGRAM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the histograms
    for (int i = 0; i < NUM_HISTOGRAMS; i++) {
        std::cout << "Histogram " << i << ": ";
        for (int j = 0; j < HISTOGRAM_BINS; j++)
            std::cout << h_histograms[i][j] << " ";
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_histograms);

    return 0;
}

