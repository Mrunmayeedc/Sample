#include <stdio.h>
#include <cuda.h>

__global__ void matrixMul(int *A, int *B, int *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        int sum = 0;

        for(int k=0;k<n;k++)
            sum += A[row*n + k] * B[k*n + col];

        C[row*n + col] = sum;
    }
}

int main()
{
    int n;

    printf("Enter matrix size (n x n): ");
    scanf("%d", &n);

    int size = n * n * sizeof(int);

    int *A = (int*)malloc(size);
    int *B = (int*)malloc(size);
    int *C = (int*)malloc(size);

    printf("Enter first matrix elements:\n");
    for(int i=0;i<n*n;i++)
        scanf("%d", &A[i]);

    printf("Enter second matrix elements:\n");
    for(int i=0;i<n*n;i++)
        scanf("%d", &B[i]);

    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid(
        (n + threadsPerBlock.x - 1)/threadsPerBlock.x,
        (n + threadsPerBlock.y - 1)/threadsPerBlock.y
    );

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result Matrix:\n");
    for(int i=0;i<n;i++) {
        for(int j=0;j<n;j++) {
            printf("%d ", C[i*n+j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
