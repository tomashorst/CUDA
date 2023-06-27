#include<thrust/device_vector.h>
#include<thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include<iostream>

struct mifuncion
{
	float a;
	mifuncion(float _a):a(_a){};
	
	__device__ __host__ 
	float operator()(float x){
		return sinf(a*x);
	}
};

int main(int arch, char **argv)
{
	int N=10000000;
	thrust::device_vector<float> y(N); // coordenada

	float a=1.0; mifuncion op(a); 
	thrust::transform(
		thrust::make_counting_iterator(0),thrust::make_counting_iterator(N),
		y.begin(),op
	); 
	// y={sin(a*0),sin(a*1),..., sin(a*(N-1))}

	std::cout << y[2] << " vs " << sin(a*2.0) << std::endl;
}
