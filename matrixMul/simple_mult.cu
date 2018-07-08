#include <stdio.h>
#include <cuda.h>

__global__
void multMats(float * A, float * B, float * C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //copy final element value to the C matrix
    if (row < m && col < k){
        float elementC = 0;
        for (int i = 0; i < n; ++i) {
            elementC += A[row*n+i]*B[i*n+col];
        }
        C[row*k+col] = elementC;
    }
}

int main(int argc, char ** argv)
{
    float *hostA;
    float *hostB;
    float *hostC;

    float *deviceA;
    float *deviceB;
    float *deviceC;

    int m = 512; // number of A rows
    int n = 512; // number of A columns (or B rows)
    int k = 512; // number of B columns

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
    int threadsPerBlock = 32;
    int n_blocks = ceil(n*n/32.0);
    printf("CUDA kernel launch with %d blocks of %d threads\n", 
        n_blocks, threadsPerBlock);

    printf("Inactive Threads %d \n", (n_blocks*threadsPerBlock)-(n*n));

    cudaEventRecord(start);
    multMats<<<n_blocks, threadsPerBlock>>>(deviceA, deviceB, deviceC, m, n, k);
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
