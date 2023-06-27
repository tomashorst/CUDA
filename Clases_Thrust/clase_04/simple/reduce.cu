#include<thrust/reduce.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<cstdio>
#include "cpu_timer.h"

int main(int argc, char **argv)
{
	thrust::host_vector<float> hx( 10000000 );

	thrust::generate(hx.begin(),hx.end(),rand);

	thrust::device_vector<float> dx=hx;

	cpu_timer reloj;

	reloj.tic();
	float sum=thrust::reduce(dx.begin(),dx.end());

	//esta instruccion solo la necesito y es valida en cuda
	#ifdef CUDA
	cudaDeviceSynchronize();
	#endif

	printf("ejecutable:%s suma=%f en %lf ms\n", argv[0], sum, reloj.tac());

	#ifdef OMP
	printf("omp threads=%d",omp_get_num_threads());
	#endif

	return 0;
}
