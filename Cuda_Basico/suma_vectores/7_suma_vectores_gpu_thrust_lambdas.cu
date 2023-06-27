// solucion thrust

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "gpu_timer.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <stdlib.h>

#define SIZE	1024

// no hay kernel!...

int main(int argc, char **argv)
{
	int N;
	
	if(argc==2) N=atoi(argv[1]);
	else N=SIZE;

	// vectores (containers) de device
	thrust::device_vector<int> d_a(N);
	thrust::device_vector<int> d_b(N);
	thrust::device_vector<int> d_c(N);

	// inicializacion arrays de device
	thrust::sequence(d_a.begin(),d_a.end());
	thrust::copy(d_a.begin(),d_a.end(),d_b.begin());

	// timer para gpu...
	gpu_timer Reloj;
	Reloj.tic();

	// suma paralela en el device
	// usando lambdas (nvcc --extended-lambda)
	thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(), 
	[=] __device__ (int i,int j) 
	{
		return (i + j);
	});	

	// milisegundos transcurridos
	printf("thrust::transform, N= %d t= %lf ms\n", N, Reloj.tac());	

	// crea un vector de host y copia el resultado del device
	thrust::host_vector<int> c(d_c);

	// verificacion del resultado
	for( int i = 0; i < N; ++i){
		//printf("c[%d] = %d\n", i, c[i]);
		assert(c[i]==2*i);
	}

	//la memoria se libera automaticamente...

	return 0;
}
