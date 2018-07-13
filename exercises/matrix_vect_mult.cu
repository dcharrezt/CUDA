#include <stdio.h>

__global__ void matVectMult(float* d_B, float* d_C, float* d_A, int numElements){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.f;
	if(row < numElements && col < numElements) {
		for(int i=0; i<numElements; i++)
			sum += d_B[row+col*i];
		d_A[row] = sum + d_C[row];		
	}
}

int main() {

	// size of matrix and vector
	int n = 512;
	size_t vectorSize = n*sizeof(float);
	size_t matrixSize = n*n*sizeof(float);

	float *h_vA;
	float *h_mB;
	float *h_vC;

	float *d_A;
	float *d_B;
	float *d_C;

	// allocate data in host
	h_vA = (float*) malloc( vectorSize );
	h_mB = (float*) malloc( matrixSize );
	h_vC = (float*) malloc( vectorSize );

	// matrix initialization
	for(int i=0; i<n; i++)
		h_vC[i] = i;
	for(int i=0; i<n*n; i++)
		h_mB[i] = i;

	// allocate data in device
	cudaMalloc((void**)&d_A, vectorSize);
	cudaMalloc((void**)&d_B, matrixSize);
	cudaMalloc((void**)&d_C, vectorSize);
	
	// copy inputs to device
	cudaMemcpy(d_C, h_vC, vectorSize ,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_mB, matrixSize ,cudaMemcpyHostToDevice);
	
	// launch kernel
	dim3 DimThreadsPerBlock(32,32,1);
	dim3 DimBlocks(ceil((n*n)/32.0),ceil(n/32.0),1);
	matVectMult<<< DimBlocks, DimThreadsPerBlock>>>(d_B,d_C,d_A, n);
	
	// copy output to host
	cudaMemcpy(h_vA, d_A, vectorSize,cudaMemcpyDeviceToHost);

	// freeing space
	free(h_vA);
	free(h_mB);
	free(h_vC);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
