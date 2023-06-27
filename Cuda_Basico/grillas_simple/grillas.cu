#include <stdio.h>

// kernel
__global__ void Quiensoy()
{
	printf("Soy el thread (%d,%d,%d) del bloque (%d,%d,%d) [blockDim=(%d,%d,%d),gridDim=(%d,%d,%d)] \n",threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z,
blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z);

}

int main()
{
	//TODO: pruebe distintas grillas	

	//ejemplo1: 4 blocks, y 3 threads/block: 
	dim3 nb(4,2,1); dim3 nt(3,1,1);		
	Quiensoy<<< nb, nt>>>();

	// espera a que los threads hayan terminado
	cudaDeviceSynchronize();
	return 0;
}



