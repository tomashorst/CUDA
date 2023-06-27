#include<stdio.h>
#include <cuda_runtime.h>
#include "gpu_timer.h"
#include "cpu_timer.h"
#include <iostream>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}


// Matrix multiplication in cpu
void MatMulcpu(Matrix A, Matrix B, Matrix C)
{
//TODO: completar para que C=A*B en la cpu
}

#define DIM	1024
#define  IDX2C(i,j,ld) (((j)*(ld))+( i ))

int main(int argc, char **argv)
{
	//TODO: completar
	// (1) alocar e inicializar A y B en host
	// (2) chequear y cronometrar MatMul
	// (3) chequear y cronometrar MatMulcpu
	// (4) Comparar CPU vs GPU para distintos tamaños de matriz
	Matrix A;
	Matrix B;
	Matrix C;
	A.width=A.height=B.width=B.height=C.height=C.width;
	A.elements=(float *)malloc(DIM*DIM*sizeof(float));
	B.elements=(float *)malloc(DIM*DIM*sizeof(float));
	C.elements=(float *)malloc(DIM*DIM*sizeof(float));
	for(int i=0;i<DIM*DIM;i++){
		B.elements[i]=rand()/RAND_MAX;
		A.elements[i]=rand()/RAND_MAX;
	}
        return 0;
}


