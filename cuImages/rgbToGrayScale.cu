#include "CImg.h"
#include  <iostream>

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
    uchar4 color = rgbaImage[index];
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
  size_t numBlocksX = ceil(numRows / threadsPerBlock);
  size_t numBlocksY = ceil(numRows / threadsPerBlock);
  const dim3 blockSize(threadsPerBlock, threadsPerBlock, 1);
  const dim3 gridSize(numBlocksX, numBlocksY, 1);
  rgba_to_greyscale<<<gridSize,blockSize>>>(d_rgbaImage,d_greyImage,numRows,numCols);
}

int main() {
	std::string input_file = "images/lena.jpg";	
	std::string output_file = "images/grey_lena.jpg";

	uchar4	*h_rgbImage, *d_rgbImage;
	unsigned char *h_greyImage, *d_greyImage;

	CImg<unsigned char> image(input_file.c_str()),
		grayWeight(image.width(), image.height(), 1, 1, 0);
	
	const int image_rows = image.width();
	const int image_cols = image.height();

// for all pixels x,y in image
  cimg_forXY(image,x,y) {

    // Separation of channels
    int R = (int)image(x,y,0,0);
    int G = (int)image(x,y,0,1);
    int B = (int)image(x,y,0,2);
    // Real weighted addition of channels for gray
    int grayValueWeight = (int)(0.299*R + 0.587*G + 0.114*B);
    // saving píxel values into image information
    grayWeight(x,y,0,0) = grayValueWeight;
}
  
  grayWeight.save( output_file.c_str() );


}

