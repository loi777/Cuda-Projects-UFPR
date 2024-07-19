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



//
__global__ void rgb2uintKernelSHM(u_int *argb, u_char *rgb, int width, int height){


}


__global__ void blurKernelSHM( u_int *argb_out, u_int *argb_in, int width, int height){

}


__global__ void uint2rgbKernelSHM(u_int *argb, u_char *rgb, int width, int height){


}



//------------------------------------


//========================================================



int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;

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

  wbTime_start(Generic, "Doing GPU Computation (memory + compute)");

  //---------------------------------------------------------------------------------------

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)d_mem_3ch, imageWidth * imageHeight * sizeof(u_char) * 3);
  cudaMalloc((void **)d_input_int, imageWidth * imageHeight * sizeof(u_int) * 3);
  cudaMalloc((void **)d_output_int, imageWidth * imageHeight * sizeof(u_int) * 3);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying 3channel data to the GPU");
  cudaMemcpy(d_mem_3ch, h_input_3ch, imageWidth * imageHeight * sizeof(u_char) * 3, cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying 3channel data to the GPU");


  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  ///////////////////////////////////////////////////////

  printf("Transforming Device 3channel into u_int\n");
  rgb2uintKernelSHM(d_input_int, d_mem_3ch, imageWidth, imageHeight);

  printf("Applying Blur effect in Device u_int memory\n");
  blurKernelSHM(d_input_int, d_output_int, imageWidth, imageHeight);

  printf("Transforming Device u_int into 3channel\n");
  uint2rgbKernelSHM(d_output_int, d_mem_3ch, imageWidth, imageHeight);


  ///////////////////////////////////////////////////////
  wbTime_stop(Compute, "Doing the computation on the GPU");
  ///////////////////////////////////////////////////////


  wbTime_start(Copy, "Copying 3channel data from the GPU");
  cudaMemcpy(h_output_3ch, d_mem_3ch, imageWidth * imageHeight * sizeof(u_char) * 3, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying 3channel data from the GPU");

  //---------------------------------------------------------------------------------------

  wbTime_stop(Generic, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);
  // DEBUG: if you want to see your image, 
  //   will generate file bellow in current directory
  //wbExport( "blurred.ppm", outputImage );

  cudaFree(d_mem_3ch);
  cudaFree(d_input_int);
  cudaFree(d_output_int);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
