// v0.2 modified by WZ

#include <stdio.h>

#include "wb4.h"  // use our lib instead (under construction)



typedef unsigned int u_int;
typedef unsigned char u_char;

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define CHECK(A, M, ...) \
  check((A), __FILE__, __LINE__, __func__, (M), ##__VA_ARGS__)

#define BLURSIZE 5

#define BLCK 4      // Block qtd
#define NTHx 16     // threads em x
#define NTHy 16     // threads em y

//========================================================



__device__ __forceinline__ void checkInsideLine(u_int* blockPosiX, u_int* blockPosiY, int width) {
  while (*blockPosiX > width) {
    *blockPosiX -= width;
    *blockPosiY += NTHy;
  }
}



//------------------------------------



// transform the device memory from u_char to u_int
__global__ void rgb2uintKernelSHM(u_int *argb, u_char *rgb, int width, int height) {
  u_int blockPosiY = 0;                              // block origin in vertical

  u_int blockPosiX = blockIdx.x * NTHx;              // block origin in horizontal
  checkInsideLine(&blockPosiX,&blockPosiY,width);


  while (blockPosiY < height) {  // while the block is inside the image
    u_int pixelPosition = ((blockPosiX + threadIdx.x)*3) + ((blockPosiY + threadIdx.y)*width*3);
    if (pixelPosition > height * (width*3)) {          // if pixel is valid
      argb[pixelPosition] = (u_int)rgb[pixelPosition];
    }

    //

    __syncthreads();

    //

    blockPosiX += NTHx * gridDim.x;
    checkInsideLine(&blockPosiX,&blockPosiY,width);
  }
}


// Apply blur effect in the device memory
// Blur logic is the simplest, a simple average of all pixels inside the
// BLURSIZE Block
__global__ void blurKernelSHM( u_int *argb_out, u_int *argb_in, int width, int height) {
  // The shared memory will be a window containing 1 pixel for 1 thread, 16x16 by default
  // plus an area of the size of BLURSIZE around this window, to allow for the correct blur.
  // remember that 1 pixel is actually 3 indexes in this array
  //__shared__ float s_imageWindow[blockDim.x * (BLURSIZE*2) * blockDim.y * (BLURSIZE*2) * 3];

  //---

  u_int blockPosiY = 0;                              // block origin in vertical

  u_int blockPosiX = blockIdx.x * NTHx;              // block origin in horizontal
  checkInsideLine(&blockPosiX,&blockPosiY,width);

  //--

  while (blockPosiY < height) {  // while the block is inside the image
    u_int pixelPosition = ((blockPosiX + threadIdx.x)*3) + ((blockPosiY + threadIdx.y)*width*3);
    if (pixelPosition > height * (width*3)) {          // if pixel is valid
      argb_out[pixelPosition] = argb_in[pixelPosition];
    }

    //

    __syncthreads();

    //

    blockPosiX += NTHx * gridDim.x;
    checkInsideLine(&blockPosiX,&blockPosiY,width);
  }
}


// transform the device memory from u_int to u_char
__global__ void uint2rgbKernelSHM(u_int *argb, u_char *rgb, int width, int height) {
  u_int blockPosiY = 0;                              // block origin in vertical

  u_int blockPosiX = blockIdx.x * NTHx;              // block origin in horizontal
  checkInsideLine(&blockPosiX,&blockPosiY,width);


  while (blockPosiY < height) {  // while the block is inside the image
    u_int pixelPosition = ((blockPosiX + threadIdx.x)*3) + ((blockPosiY + threadIdx.y)*width*3);
    if (pixelPosition > height * (width*3)) {          // if pixel is valid
      rgb[pixelPosition] = (u_char)argb[pixelPosition];
    }

    //

    __syncthreads();

    //

    blockPosiX += NTHx * gridDim.x;
    checkInsideLine(&blockPosiX,&blockPosiY,width);
  }
}



//========================================================



int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;

  dim3 blockDimnension = dim3(NTHx, NTHy);

  u_char *h_input_3ch;  // host memory
  u_char *h_output_3ch; //

  u_char *d_mem_3ch;    // device input memory
  u_int *d_input_int;   // device input int memory
  u_int *d_output_int;  // device output memory

  //--------
  
  // USER INPUT PARAMETERS
  args = wbArg_read(argc, argv);
  inputImageFile = wbArg_getInputFile(args, 1);
  printf( "imagem de entrada: %s\n", inputImageFile );

  //  INPUT IMAGE
  inputImage = wbImport(inputImageFile);
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  h_input_3ch = wbImage_getData(inputImage);   // save unsigned char value of every pixel

  // OUTPUT IMAGE
  outputImage = wbImage_new(imageWidth, imageHeight, 3);
  h_output_3ch = wbImage_getData(outputImage); // save unsigned char value of every pixel

  // GET GENERAL VARIABLES
  u_int imageArraySize = (imageWidth*3) * imageHeight; //Substitute: (imageWidth*3) * imageHeight

  wbTime_start(Generic, "Doing GPU Computation (memory + compute)");

  //---------------------------------------------------------------------------------------
  
  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&d_mem_3ch, imageArraySize * sizeof(u_char));
  cudaMalloc((void **)&d_input_int, imageArraySize * sizeof(u_int));
  cudaMalloc((void **)&d_output_int, imageArraySize * sizeof(u_int));
  wbTime_stop(GPU, "Doing GPU memory allocation");
  
  wbTime_start(Copy, "Copying 3channel data to the GPU");
  cudaMemcpy(d_mem_3ch, h_input_3ch, imageArraySize * sizeof(u_char), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying 3channel data to the GPU");
  

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  ///////////////////////////////////////////////////////

  
  printf("Transforming Device 3channel into u_int\n");
  rgb2uintKernelSHM<<<BLCK, blockDimnension>>>(d_input_int, d_mem_3ch, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  cudaMemcpy(h_output_3ch, d_input_int, imageArraySize * sizeof(u_char), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 23; i++) {
    printf("intPixel[%d] = %d %d %d   charPixel[%d] = %d %d %d\n", i, h_output_3ch[i], h_output_3ch[i+1], h_output_3ch[i+2],
                                                                   i, h_input_3ch[i], h_input_3ch[i+1], h_input_3ch[i+2]);
  }

  //printf("Applying Blur effect in Device u_int memory\n");
  //blurKernelSHM<<<BLCK, blockDimnension>>>(d_input_int, d_output_int, imageWidth, imageHeight);
  //cudaDeviceSynchronize();

  //printf("Transforming Device u_int into 3channel\n");
  //uint2rgbKernelSHM<<<BLCK, blockDimnension>>>(d_output_int, d_mem_3ch, imageWidth, imageHeight);
  //cudaDeviceSynchronize();


  ///////////////////////////////////////////////////////
  wbTime_stop(Compute, "Doing the computation on the GPU");
  ///////////////////////////////////////////////////////


  wbTime_start(Copy, "Copying 3channel data from the GPU");
  cudaMemcpy(h_output_3ch, d_mem_3ch, imageArraySize * sizeof(u_char), cudaMemcpyDeviceToHost);

  // copy result obtained from GPU to outputImage.data
  for (int i = 0; i < imageArraySize; i+= 3) {
    outputImage.data[i] = h_output_3ch[i];
    outputImage.data[i+1] = h_output_3ch[i+1];
    outputImage.data[i+2] = h_output_3ch[i+2];
  }

  wbTime_stop(Copy, "Copying 3channel data from the GPU");

  //---------------------------------------------------------------------------------------

  wbTime_stop(Generic, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);
  // DEBUG: if you want to see your image, 
  //   will generate file bellow in current directory
  wbExport( "blurred.ppm", outputImage );

  cudaFree(d_mem_3ch);
  cudaFree(d_input_int);
  cudaFree(d_output_int);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
