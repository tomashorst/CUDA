// Tabulador de funcion

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "gpu_timer.h"
#include "cpu_timer.h"

#define SIZE	1024

// funcion para rellenar
__host__ __device__ float Mifuncion(int i)
{
	return tanh(cos(exp(-i*0.01)+0.02));
}


// kernel para tabular
__global__ void Llenar(float *a)
{
	// indice de thread mapeado a indice de array 
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	a[i]=Mifuncion(i);
}

int main(int argc, char **argv)
{
	int N;
	
	if(argc==2) N=atoi(argv[1]);
	else N=SIZE;

	// punteros a memoria de host
	float *a;

	// punteros a memoria de device
	float *d_a; 

	// alocacion memoria de host
	a = (float *)malloc(N*sizeof(float));

	// alocacion memoria de device
	cudaMalloc( &d_a, N*sizeof(float));

	// grilla de threads suficientemente grande...
	dim3 nThreads(256);
	dim3 nBlocks((N + nThreads.x - 1) / nThreads.x);

	// timer para gpu...
	gpu_timer RelojGPU;
	RelojGPU.tic();

	// llena en el device
	Llenar<<< nBlocks, nThreads >>>(d_a);

	// milisegundos transcurridos
	printf("N= %d \t t_GPU= %lf ms,\t", N, RelojGPU.tac());	
	
	// copia (solo del resultado) del device a host
	cudaMemcpy( a, d_a, N*sizeof(float), cudaMemcpyDeviceToHost );

	float tolerancia=0.0000001;
	float value;
	for(int i=0;i<N;i++){
		value = Mifuncion(i);
		if(fabs(a[i]-value)>tolerancia)
		{
			printf("%f VS %f\n", a[i], value);
			exit(1);
		}
	}

	// timer para cpu...
	cpu_timer RelojCPU;
	RelojCPU.tic();

	// llena en el device
	for(int i=0;i<N;i++) a[i]=Mifuncion(i);

	// milisegundos transcurridos
	printf("t_CPU= %lf ms\n", RelojCPU.tac());	


	// liberacion memoria de host y device
	free(a);
	cudaFree(d_a);

	return 0;
}
