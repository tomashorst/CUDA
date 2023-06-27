#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/functional.h>
#include<thrust/transform_reduce.h>
#include<cstdlib>
#include<iostream>
#include<cstdio>
 
using namespace thrust::placeholders;

int main(int arch, char **argv)
{
	srand(13);
	int N=10000000;
	thrust::host_vector<float> hvec(N);
	thrust::generate(hvec.begin(),hvec.end(),rand);

	thrust::device_vector<float> dvec(N);
	thrust::copy(hvec.begin(),hvec.end(),dvec.begin());

	//thrust::transform(dvec.begin(),dvec.end(),dvec.begin(),_1/RAND_MAX);
	//float mayor = thrust::reduce(dvec.begin(),dvec.end(),-1.f,thrust::maximum<float>());

	float mayor = 
	transform_reduce(dvec.begin(),dvec.end(),_1/RAND_MAX,-1.f,thrust::maximum<float>());
	
	//std::cout << mayor << std::endl; 
	printf("%.10f\n",mayor);	
}
