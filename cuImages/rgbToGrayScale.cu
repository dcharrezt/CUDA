#include <stdio.h>

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                      unsigned char* const greyImage,
                      int numRows, int numCols)
{
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  if( y<numCols && x < numRows) {
    int index = numRows*y + x;
    uchar4 color = rgbaImage[image];
    unsigned char grey = (unsigned char)(0.299f*color.x + 0.587f*color.y +
                                          0.114f*color.z);
    greyImage[index] = grey;
  }
}

void your_rgba_to_greyscale(const uchar4* const h_rgbaImage,
                            uchar4* const d_rgbaImage,
                            unsigned char* const d_greyImage,
                            size_t numRows, size_t numCols)
{
  size_t threadsPerBlock = 32;
  size_t numBlocksX = ceil(numRows / blockWidth);
  size_t numBlocksY = ceil(numRows / blockWidth);
  const dim3 blockSize(threadsPerBlock, threadsPerBlock, 1);
  const dim3 gridSize(numBlocksX, numBlocksY, 1);
  rgba_to_greyscale<<<gridSize,blockSize>>>(d_rgbaImage,d_greyImage,numRows,numCols);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
