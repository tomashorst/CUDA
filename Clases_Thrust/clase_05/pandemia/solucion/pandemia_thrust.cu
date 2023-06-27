#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <iostream>
#include <cstdio>

#define Sus	0
#define Inf	1
#define Rec	2
#define Beta	3

#define Gamma	0.1
#define Dt	0.1

struct arbitrary_functor1
{
    template <typename Tuple>
    __device__
    thrust::tuple<float,float,float> operator()(Tuple t) //:public thrust::unary_function<Tuple,Tuple>
    {
	float newS, newI, newR;
	float beta = thrust::get<Beta>(t);

        // newS[i] = S[i] + Dt*(-S[i]*I[i]*beta);
	newS = 	thrust::get<Sus>(t) + Dt*(-thrust::get<Sus>(t)*thrust::get<Inf>(t)*beta);

        // newI[i] = I[i] + Dt*(S[i]*I[i]*beta - I[i]*gamma);
	newI = 	thrust::get<Inf>(t) + Dt*(thrust::get<Sus>(t)*thrust::get<Inf>(t)*beta - thrust::get<Inf>(t)*Gamma);

        // newR[i] = R[i] + Dt*(gamma*I[i]);	
	newR = 	thrust::get<Rec>(t) + Dt*(thrust::get<Inf>(t)*Gamma);

	//printf("new => %f %f %f\n",newS, newI, newR);

	return thrust::make_tuple(newS, newI, newR);
    }
};

int main(void)
{
    // allocate storage
    thrust::device_vector<float> S(10);
    thrust::device_vector<float> I(10);
    thrust::device_vector<float> R(10);
    thrust::device_vector<float> beta(10);

    // initialize input vectors
    thrust::fill(S.begin(),S.end(),1.0);	
    thrust::fill(I.begin(),I.end(),0.001);	
    thrust::fill(R.begin(),R.end(),0.0);	
    thrust::sequence(beta.begin(),beta.end(),0.02,0.02);	

    //thrust::copy(beta.begin(), beta.end(), std::ostream_iterator<float>(std::cout, "\t"));
    //std::cout << "\n";

    for(int n=0;n<5000;n++){	
	thrust::copy(I.begin(), I.end(), std::ostream_iterator<float>(std::cout, "\t"));
	std::cout << "\n";

    	// apply the transformation
    	thrust::transform
	(
		     thrust::make_zip_iterator( thrust::make_tuple(S.begin(), I.begin(), R.begin(),beta.begin()) ),
                     thrust::make_zip_iterator( thrust::make_tuple(S.end(), I.end(), R.end(),beta.end()) ),
		     thrust::make_zip_iterator( thrust::make_tuple(S.begin(), I.begin(), R.begin()) ),
                     arbitrary_functor1()
	);
    }	
}

