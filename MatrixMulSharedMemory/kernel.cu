
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 16
#define TILE_SIZE 4

__global__ void matrixMultiplication(int* a, int* b, int* c, int n) {

    __shared__ int s_a[TILE_SIZE][TILE_SIZE];
    __shared__ int s_b[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    int sum = 0;
    for (int i = 0; i < n / TILE_SIZE; i++) {
        s_a[ty][tx] = a[row * n + i * TILE_SIZE + tx];
        s_b[ty][tx] = b[(i * TILE_SIZE + ty) * n + col];
        __syncthreads();
        for (int j = 0; j < TILE_SIZE; j++) {
            sum += s_a[ty][j] * s_b[j][tx];
        }
        __syncthreads();
    }
    c[row * n + col] = sum;
}

void printMatrix(int* m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", m[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {

    int a[N * N], b[N * N], c[N * N], d[N * N];
    int* dev_a, * dev_b, * dev_c;

    cudaMalloc((void**)&dev_a, N * N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * N * sizeof(int));

    for (int i = 0; i < N * N; i++) {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
        d[i] = 0;
    }

    printf("Input matrices:\n");
    printf("Matrix A:\n");
    printMatrix(a, N);
    printf("Matrix B:\n");
    printMatrix(b, N);

    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(N / TILE_SIZE, N / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    matrixMultiplication << <grid, block >> > (dev_a, dev_b, dev_c, N);

    cudaMemcpy(c, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Output matrix:\n");
    printMatrix(c, N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                d[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }

    printf("Expected output matrix:\n");
    printMatrix(d, N);

    for (int i = 0; i < N * N; i++) {
        if (c[i] != d[i]) {
            printf("Error: matrix multiplication result does not match expected result.\n");
            break;
        }
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

