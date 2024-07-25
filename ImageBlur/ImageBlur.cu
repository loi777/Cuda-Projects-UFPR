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


#define BLURSIZE 5

#define BLCK 4      // Block qtd
#define NTHx 16     // threads em x
#define NTHy 16     // threads em y

#define WINDOWX ((NTHx + (BLURSIZE*2))*3)   // horizontal size of our working window for blur
#define WINDOWY (NTHy + (BLURSIZE*2))   // vertical size of our working window for blur



//========================================================



// keeps block indexes inside the image, if it overflows horizontally
// send it down one block line and reset horizontal position.
__device__ __forceinline__ void checkInsideLine(u_int* blockPosiX, u_int* blockPosiY, int width, int height) {
  while (*blockPosiX >= width && *blockPosiY <= height) {
    // only do this if outside horizontal image size
    // and inside vertical image size

    *blockPosiX = blockIdx.x * NTHx;    // reset horizontal position
    *blockPosiY += NTHy;                // next vertical position
  }
}


__device__ __forceinline__ u_int pixelToArray(int x, int y, int width) {
  return ((x*3) + (y*width*3));
}



//------------------------------------



// translate the unified array into a 2d array for our blurry window
// SyncThreads at the end of function
__device__ __forceinline__ void getWorkWindow(u_int s_imageWindow[WINDOWY][WINDOWX], u_int* argb_in, u_int blockPosiX, u_int blockPosiY, int width, int height) {

  // although many times slower, only thread 0.0 collects the shared memory, for simplicity sake
  if (threadIdx.x == 0 && threadIdx.y == 0) {

    for (int j = 0; j < WINDOWY; j++) {
      for (int i = 0; i < WINDOWX; i += 3) {
        int X = ((blockPosiX - BLURSIZE)*3) + i;  // position in real array
        int Y = blockPosiY - BLURSIZE + j;  // position in real array

        s_imageWindow[j][i] = 0;    // reset shared mem value to avoid memory trash
        s_imageWindow[j][i+1] = 0;
        s_imageWindow[j][i+2] = 0;

        if (X >= 0 && X < (width*3)) {
          if (Y >= 0 && Y < height) {    // do not collect to shared mem if outside Image
            u_int pixelPosi = X + (Y*width*3);

            s_imageWindow[j][i] = argb_in[pixelPosi];
            s_imageWindow[j][i+1] = argb_in[pixelPosi+1];
            s_imageWindow[j][i+2] = argb_in[pixelPosi+2];
          }
        }
      }
    }
  }

  // finaly sync threads to continue
  __syncthreads();
}


__device__ __forceinline__ void blurPixel(u_int* argb_out, u_int s_imageWindow[WINDOWY][WINDOWX], u_int pixelPosition) {
  int pixelsAdded = 0;                    // counter to know how many pixels were added

  int originX = (threadIdx.x + BLURSIZE)*3;   // position in shared window 
  int originY = threadIdx.y + BLURSIZE;   // position in shared window

  for (int j = -BLURSIZE; j < (BLURSIZE+1); j++) {
    for (int i = -BLURSIZE*3; i < ((BLURSIZE*3)+3); i += 3) {

      argb_out[pixelPosition] += s_imageWindow[originY + j][originX + i];
      argb_out[pixelPosition+1] += s_imageWindow[originY + j][originX + i+1];
      argb_out[pixelPosition+2] += s_imageWindow[originY + j][originX + i+2];

      if ((s_imageWindow[originY+j][originX+i] + s_imageWindow[originY+j][originX+i+1] + s_imageWindow[originY+j][originX+i+2]) > 0) {
        // if there was a pixel added this loop, we count for the average
        pixelsAdded++;
      }
    }
  }
  
  argb_out[pixelPosition] /= pixelsAdded;
  argb_out[pixelPosition+1] /= pixelsAdded;
  argb_out[pixelPosition+2] /= pixelsAdded;
}



//------------------------------------



// transform the device memory from u_char to u_int
__global__ void rgb2uintKernelSHM(u_int *argb, u_char *rgb, int width, int height) {
  u_int blockPosiY = 0;                              // block origin in vertical

  u_int blockPosiX = blockIdx.x * NTHx;              // block origin in horizontal
  checkInsideLine(&blockPosiX,&blockPosiY,width,height);


  while (blockPosiY < height) {  // while the block is inside the image

    if (blockPosiX + threadIdx.x < width && blockPosiY + threadIdx.y < height) {
      // if pixel is inside image

      u_int pixelPosition = pixelToArray(blockPosiX + threadIdx.x, blockPosiY + threadIdx.y, width);

      argb[pixelPosition] = (u_int)rgb[pixelPosition];
      argb[pixelPosition+1] = (u_int)rgb[pixelPosition+1];
      argb[pixelPosition+2] = (u_int)rgb[pixelPosition+2];

      //if (threadIdx.x == 0 && threadIdx.y == 0) printf("Copied to index: %d  |%d|%d|%d\n", pixelPosition, 
      //                                                  argb[pixelPosition], argb[pixelPosition+1], argb[pixelPosition+2]);
    }

    //

    __syncthreads();

    //

    blockPosiX += NTHx * gridDim.x;         // move block by amount of blocks
    checkInsideLine(&blockPosiX,&blockPosiY,width,height);
  }
}


// transform the device memory from u_int to u_char
__global__ void uint2rgbKernelSHM(u_int *argb, u_char *rgb, int width, int height) {
  u_int blockPosiY = 0;                              // block origin in vertical

  u_int blockPosiX = blockIdx.x * NTHx;              // block origin in horizontal
  checkInsideLine(&blockPosiX,&blockPosiY,width,height);


  while (blockPosiY < height) {  // while the block is inside the image

    if (blockPosiX + threadIdx.x < width && blockPosiY + threadIdx.y < height) {
      // if pixel is inside image

      u_int pixelPosition = pixelToArray(blockPosiX + threadIdx.x, blockPosiY + threadIdx.y, width);

      rgb[pixelPosition] = (u_char)argb[pixelPosition];
      rgb[pixelPosition+1] = (u_char)argb[pixelPosition+1];
      rgb[pixelPosition+2] = (u_char)argb[pixelPosition+2];

      //if (threadIdx.x == 0 && threadIdx.y == 0) printf("Copied to index: %d  |%d|%d|%d\n", pixelPosition, 
      //                                                  argb[pixelPosition], argb[pixelPosition+1], argb[pixelPosition+2]);
    }

    //

    __syncthreads();

    //

    blockPosiX += NTHx * gridDim.x;         // move block by amount of blocks
    checkInsideLine(&blockPosiX,&blockPosiY,width,height);
  }
}



//===========================================================================================



// Apply blur effect in the device memory
// Blur logic is the simplest, a simple average of all pixels inside the
// BLURSIZE Block
__global__ void blurKernelSHM(u_int *argb_out, u_int *argb_in, int width, int height) {
  // The shared memory will be a window containing 1 pixel for 1 thread, 16x16 by default
  // plus an area of the size of BLURSIZE around this window, to allow for the correct blur.
  // remember that 1 pixel is actually 3 indexes in this array
  __shared__ u_int s_imageWindow[WINDOWY][WINDOWX]; // s_image[y][x]
  
  //--
  
  u_int blockPosiY = 0;                              // block origin in vertical

  u_int blockPosiX = blockIdx.x * NTHx;              // block origin in horizontal
  checkInsideLine(&blockPosiX,&blockPosiY,width,height);

  //--

  getWorkWindow(s_imageWindow, argb_in, blockPosiX, blockPosiY, width, height);

  //--

  while (blockPosiY < height) {  // while the block is inside the image

    if (blockPosiX + threadIdx.x < width && blockPosiY + threadIdx.y < height) {
      // if central pixel is inside image

      u_int pixelPosition = pixelToArray(blockPosiX + threadIdx.x, blockPosiY + threadIdx.y, width);  // central pixel
      blurPixel(argb_out, s_imageWindow, pixelPosition);
    }

    //

    blockPosiX += NTHx * gridDim.x;         // move block by amount of blocks
    checkInsideLine(&blockPosiX,&blockPosiY,width,height);

    //

    __syncthreads();
    getWorkWindow(s_imageWindow, argb_in, blockPosiX, blockPosiY, width, height);
  }
}



//===========================================================================================



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
  u_int imageArraySize = (imageWidth*3) * imageHeight; //Expands to: (imageWidth*3) * imageHeight

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

  
  printf("DEVICE:  3channel => u_int\n");
  rgb2uintKernelSHM<<<BLCK, blockDimnension>>>(d_input_int, d_mem_3ch, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  printf("DEVICE:  Blur u_int memory\n");
  blurKernelSHM<<<BLCK, blockDimnension>>>(d_output_int, d_input_int, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  printf("DEVICE:  u_int => 3channel\n");
  uint2rgbKernelSHM<<<BLCK, blockDimnension>>>(d_output_int, d_mem_3ch, imageWidth, imageHeight);
  cudaDeviceSynchronize();


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
