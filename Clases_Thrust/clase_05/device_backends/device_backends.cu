#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/reduce.h>
#include<cstdlib>
#include<cstdio>
#include<thrust/functional.h>
#include "simple_timer.h"

int main(int argc, char **argv)
{
	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
	int dev; cudaGetDevice(&dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	gpu_timer cronocon;
	gpu_timer cronosin;

	#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
	printf("OMP version, with %d threads\n", omp_get_max_threads());
	omp_timer cronocon;
	omp_timer cronosin;

	#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CPP
	printf("CPP version\n");
	cpu_timer cronocon;
	cpu_timer cronosin;
	#endif

	unsigned long long N;
	if(argc==2) N=atoi(argv[1]); else N=1000000;
 
	thrust::host_vector<float> hx( N );
	thrust::generate(hx.begin(),hx.end(),rand);

	cronocon.tic();
	thrust::device_vector<float> dx=hx; 

	cronosin.tic();
	float init=dx[0];
	float suma = thrust::reduce(dx.begin(),dx.end(),init,thrust::minimum<float>());	

	float mscon = cronocon.tac();
	float mssin = cronosin.tac();

	printf("minimo entre %d numeros =%f, ms_con=%f, ms_sin=%f\n",N, suma,mscon,mssin);	

	return 0;
}
