#include <stdio.h>
#include <cuda.h>
#define TILE_DIM 16

__global__
void multMats(float * A, float * B, float * C, int m, int n, int k)
{
    //Create 2 tiles for matrix A and B at the shared memory
    __shared__ float ATile[TILE_DIM][TILE_DIM];
    __shared__ float BTile[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int thrX = threadIdx.x;
    int thrY = threadIdx.y;

    //to accumulate partial values of each element in C
    float elementC = 0;

    for (int t = 0; t < (n-1)/TILE_DIM +1; ++t)
    {
        //threads to load matrix A to shared memory
        if(row < m && t*TILE_DIM+thrX < n)
            ATile[thrY][thrX] = A[row*n + t*TILE_DIM+thrX];
        else
            ATile[thrY][thrX] = 0.0f;

        //threads to load matrix B to shared memory
        if (t*TILE_DIM+thrY < n && col < k)
            BTile[thrY][thrX] = B[(t*TILE_DIM+thrY)*k + col];
        else
            BTile[thrY][thrX] = 0.0f;

        __syncthreads();

        //calculate a partial value of thread element in C
        for (int i = 0; i < TILE_DIM; ++i)
            elementC += ATile[thrY][i] * BTile[i][thrX];

        __syncthreads();

    }
    //copy final element value to the C matrix
    if (row < m && col < k)
        C[row*k+col] = elementC;

}

int main(int argc, char ** argv)
{
    float *hostA;
    float *hostB;
    float *hostC;

    float *deviceA;
    float *deviceB;
    float *deviceC;

    int m = 1024; // number of A rows
    int n = 1024; // number of A columns (or B rows)
    int k = 1024; // number of B columns

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //allocate data in host
    hostA = (float *) malloc(m * n * sizeof(float));
    hostB = (float *) malloc(n * k * sizeof(float));
    hostC = (float *) malloc(m * k * sizeof(float));

   for (int i = 0; i < m*n; i++)//Matrix Initialization
        hostA[i]=1.0;
    for (int i = 0; i < n*k; i++)
        hostB[i]=1.0;

    //allocate data in device
    cudaMalloc((void **) &deviceA, m * n * sizeof(float));
    cudaMalloc((void **) &deviceB, n * k * sizeof(float));
    cudaMalloc((void **) &deviceC, m * k * sizeof(float));

    //copy inputs to device
    cudaMemcpy(deviceA, hostA, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, n * k * sizeof(float), cudaMemcpyHostToDevice);

    //device kernal
    dim3 DimGrid((k-1)/TILE_DIM+1, (m-1)/TILE_DIM+1, 1);
    dim3 DimBlock(TILE_DIM, TILE_DIM, 1);
    cudaEventRecord(start);
    multMats<<<DimGrid,DimBlock>>>(deviceA, deviceB, deviceC, m, n, k);
    cudaThreadSynchronize();
    cudaEventRecord(stop);

    //copy result back to host
    cudaMemcpy(hostC, deviceC, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Blocks CPU execution until stop has been recorded
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed Time: %f milliseconds\n", milliseconds);

    // Destroying events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //deallocate device
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    //deallocate host
    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
