#include "CImg.h"
#include  <iostream>

using namespace cimg_library;
using namespace std;

__global__
void rgba_to_greyscale(uchar4* rgbaImage,
                      unsigned char* greyImage,
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

void your_rgba_to_greyscale(uchar4* d_rgbaImage,
                            unsigned char* const d_greyImage,
                            size_t numRows, size_t numCols)
{
  size_t threadsPerBlock = 32;
  size_t numBlocksX = ceil(numRows / 32.0);
  size_t numBlocksY = ceil(numRows / 32.0);
  const dim3 blockSize(threadsPerBlock, threadsPerBlock, 1);
  const dim3 gridSize(numBlocksX, numBlocksY, 1);
	cout << "ads" <<endl;
  rgba_to_greyscale<<<gridSize,blockSize>>>(d_rgbaImage,d_greyImage,numRows,numCols);
	cout << "ads" <<endl;

}

int main() {
	std::string input_file = "images/lena.jpg";	
	std::string output_file = "images/grey_lena.jpg";

	uchar4	*h_rgbImage, *d_rgbImage;
	unsigned char *h_greyImage, *d_greyImage;

	CImg<unsigned char> image(input_file.c_str()),
		grayCU(image.width(), image.height(), 1, 1, 0),
		grayWeight(image.width(), image.height(), 1, 1, 0);
	
	const int image_rows = image.width();
	const int image_cols = image.height();

	h_rgbImage = (uchar4*)malloc(image_rows*image_cols*sizeof(uchar4));
	h_greyImage = (unsigned char*)malloc(image_rows*image_cols*sizeof(unsigned char));	

// for all pixels x,y in image
  cimg_forXY(image,i,j) {

    // Separation of channels
    int R = (int)image(i,j,0,0);
    int G = (int)image(i,j,0,1);
    int B = (int)image(i,j,0,2);
    // Real weighted addition of channels for gray
	h_rgbImage[image_rows*i+j].x = R;
	h_rgbImage[image_rows*i+j].y = G;
	h_rgbImage[image_rows*i+j].z = B;

    int grayValueWeight = (int)(0.299*R + 0.587*G + 0.114*B);
    // saving píxel values into image information
    grayWeight(i,j,0,0) = grayValueWeight;
}

	grayWeight.save( output_file.c_str() );
  
	cudaMalloc((void**)&d_rgbImage, image_rows*image_cols*sizeof(uchar4));
	cudaMalloc((void**)&d_greyImage, image_rows*image_cols*sizeof(unsigned char));

	cudaMemcpy(d_rgbImage, h_rgbImage, image_rows*image_cols*sizeof(uchar4),cudaMemcpyHostToDevice);

	 your_rgba_to_greyscale(d_rgbImage, d_greyImage,image_rows, image_cols);

	cudaMemcpy(h_greyImage, d_greyImage, image_rows*image_cols*sizeof(uchar4),cudaMemcpyDeviceToHost);	

cimg_forXY(image,i,j) {
	grayCU(i,j,0,0) = (unsigned char)h_greyImage[image_rows*i+j];
}
	
  grayCU.save( output_file.c_str() );


}

