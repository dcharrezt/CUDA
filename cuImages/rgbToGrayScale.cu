#include <stdio.h>
#include <CImg.h>
#include <iostream>
#include <string>

using namespace cimg_library;
using namespace std;

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

int main() {
	std::string input_file = "images/lena.jpg"	
	std::string output_file = "images/grey_lena.jpg"

	uchar4				*h_rgbImage, *d_rgbImage;
	unsigned char *h_greyImage, *d_greyImage;

	CImg<unsigned char> image(input_file),
		grayWeight(image.width(), image.height(), 1, 1, 0);
	image_rows = image.width();
	image_cols = image.height();


// for all pixels x,y in image
  cimg_forXY(image,x,y) {
    imgR(x,y,0,0) = image(x,y,0,0),    // Red component of image sent to imgR
    imgG(x,y,0,1) = image(x,y,0,1),    // Green component of image sent to imgG
    imgB(x,y,0,2) = image(x,y,0,2);    // Blue component of image sent to imgB:

    // Separation of channels
    int R = (int)image(x,y,0,0);
    int G = (int)image(x,y,0,1);
    int B = (int)image(x,y,0,2);
    // Arithmetic addition of channels for gray
    int grayValue = (int)(0.33*R + 0.33*G + 0.33*B);
    // Real weighted addition of channels for gray
    int grayValueWeight = (int)(0.299*R + 0.587*G + 0.114*B);
    // saving píxel values into image information
    gray(x,y,0,0) = grayValue;
    grayWeight(x,y,0,0) = grayValueWeight;
}
  
  grayWeight.save( "greys.jpg" );

	
  

}

