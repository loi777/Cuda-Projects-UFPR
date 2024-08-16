// returns the size of the number group of each bin
// needs some strange calculations due to precision error
u_int H_getBinSize(u_int min, u_int max, u_int segCount);

// The um array do device, obtem o minimo e o maximo
void H_getDeviceMinMax(u_int* d_array, u_int start, u_int size, u_int* h_min, u_int* h_max);


// Kernel para calcular histogramas em particoes
// Cada bloco eh responsavel por um histograma (linha da matriz)
__global__ void H_getHistogram(u_int *HH, u_int *Hg, u_int h, u_int *Input, u_int nElements, u_int nMin, u_int nMax, u_int segSize, u_int binWidth);

// calculates the scan of the global histogram and saves it into the horizontal scan
__global__ void H_horizontalScan(u_int *Hg, u_int *SHg, u_int h);

// calculates the scan of each non-global histogram, saving it in different lines of the vertical scan
__global__ void H_verticalScan(u_int *HH, u_int *PSv, u_int h);

// Uses the consultation table to separate the groups of numbers according to their bins
// saves in output device memory
__global__ void H_Partitioner(u_int *HH, u_int *SHg, u_int *PSv, u_int h, u_int *input, u_int* output, u_int nElements, u_int nMin, u_int nMax, u_int segSize, u_int binWidth);
