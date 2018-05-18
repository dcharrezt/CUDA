#include <stdio.h>
#include <cuda.h>
#include <math.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {

    // Print the vector length to be used, and compute its size
    int numElements = strtol(argv[1], NULL, 10);
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Variables to do the timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate the host vectors
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device vectors
    float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);

    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);

    float *d_C = NULL;
    cudaMalloc((void **)&d_C, size);

    // Copy the host vectors to the device vectors
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int n_blocks = ceil(numElements/256.0);
    printf("CUDA kernel launch with %d blocks of %d threads\n", 
        n_blocks, threadsPerBlock);

    printf("Inactive Threads %d \n", (n_blocks*threadsPerBlock)-numElements);

    cudaEventRecord(start);
    vectorAdd<<< n_blocks, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaEventRecord(stop);

    // Copy the device result vector to host result vector
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Blocks CPU execution until stop has been recorded
    cudaEventSynchronize(stop);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            printf("Failed at", i);
        }
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed Time: %f milliseconds\n", milliseconds);

    // Destroying events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}
