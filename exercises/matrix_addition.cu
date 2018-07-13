#include <stdio.h>

__global__ void addition(float* d_A, float* d_B, float* d_C, int numElements){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float tmp_1, tmp_2;
	if(row < numElements && col < numElements) {
		tmp_1 = d_A[col];
		__syncthreads();
		tmp_2 = d_B[col];
		__syncthreads();
		d_C[row] = tmp_1 + tmp_2;
		__syncthreads();
	}
}

__global__
void additionByRows(float* d_A, float* d_B, float* d_C, int numRows) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
	float tmp_1, tmp_2;
	if(row<numRows && col<numRows){
		for(int i=0; i<numRows; i++){
			tmp_1 = d_A[row+i*col];
                	__syncthreads();
                	tmp_2 = d_B[row+i*col];
                	__syncthreads();
               		d_C[row] = tmp_1 + tmp_2;
                	__syncthreads();		
		}
	}
}

__global__
void additionByColumns(float* d_A, float* d_B, float* d_C, int numColumns) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float tmp_1, tmp_2;
        if(row<numColumns && col<numColumns){
                for(int i=0; i<numColumns; i++){
                        tmp_1 = d_A[row*i+col];
                        __syncthreads();
                        tmp_2 = d_B[row*i+col];
                        __syncthreads();
                        d_C[row] = tmp_1 + tmp_2;
                        __syncthreads();
                }
        }
}

int main() {

	// size of matrixes
	int numArows = 512;
	int numAcolumns = 512;
	int numBcolumns = 512;

	float *h_A;
	float *h_B;
	float *h_C;

	float *d_A;
	float *d_B;
	float *d_C;

	// allocate data in host
	h_A = (float*) malloc( numArows*numAcolumns*sizeof(float));
	h_B = (float*) malloc( numAcolumns*numBcolumns*sizeof(float));
	h_C = (float*) malloc( numArows*numBcolumns*sizeof(float));

	// matrix initialization
	for(int i=0; i<numArows*numAcolumns; i++)
		h_A[i] = i;
	for(int i=0; i<numAcolumns*numBcolumns; i++)
		h_B[i] = i;

	// allocate data in device
	cudaMalloc((void**)&d_A, numArows*numAcolumns*sizeof(float));
	cudaMalloc((void**)&d_B, numAcolumns*numBcolumns*sizeof(float));
	cudaMalloc((void**)&d_C, numArows*numBcolumns*sizeof(float));
	
	// copy inputs to device
	cudaMemcpy(d_A, h_A, numArows*numAcolumns*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, numAcolumns*numBcolumns*sizeof(float),cudaMemcpyHostToDevice);
	
	// launch kernel
	int threadsPerBlock = 32;
	int n_blocks = ceil(numAcolumns*numAcolumns/32.0);
	
	addition<<<n_blocks, threadsPerBlock>>>(d_A,d_B,d_C, numAcolumns*numAcolumns);
	
	// copy output to host
	cudaMemcpy(h_C, d_C, numArows*numBcolumns*sizeof(float),cudaMemcpyDeviceToHost);

	// freeing space
	free(h_A);
	free(h_B);
	free(h_C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
