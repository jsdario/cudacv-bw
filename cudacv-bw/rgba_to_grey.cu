#include "utils.h"
#include <stdio.h>

__global__
void rgba_to_grey_kernel(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  //TODO
  //Fill in the kernel to convert from color to greyscale
  //the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;
  //Note: We will be ignoring the alpha channel for this conversion
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uchar4 px = rgbaImage[i]; // thread pixel to process
  greyImage[i] = .299f * px.x +
                 .587f * px.y +
                 .114f * px.z;
    
  //First create a mapping from the 2D block and grid locations
  //to an absolute 2D location in the image, then use that to
  //calculate a 1D offset
}

void rgba_to_grey(uchar4 * const d_rgbaImage,
                  unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched
  const dim3 blockSize(512, 1, 1);  //TODO
  const dim3 gridSize(512, 1, 1);  //TODO
  rgba_to_grey_kernel<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}