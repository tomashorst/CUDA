#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <thrust/tuple.h>

// This example shows how to implement an arbitrary transformation of
// the form output[i] = F(first[i], second[i], third[i], ... ).
// In this example, we use a function with 3 inputs and 1 output.
//
// In this example, we can implement the transformation,
//      D[i] = A[i] + B[i] * C[i];
//
// by passing raw pointers to the functor constructor
//
// Note that we could extend this example to implement functions with an
// arbitrary number of input arguments by zipping more sequence together.
// With the same approach we can have multiple *output* sequences, if we 
// wanted to implement something like
//      D[i] = A[i] + B[i] * C[i];
//      E[i] = A[i] + B[i] + C[i];
//
// The possibilities are endless! :)


struct arbitrary_functor
{       

    float *s[20];
    arbitrary_functor(float *a1,float *a2,float *a3, float *a4){
	s[0]=a1; s[1]=a2; s[2]=a3; s[3]=a4;
    }	

    __host__ __device__
    void operator()(int i)
    {
     	// D[i] = A[i] + B[i] * C[i];
     	//thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) * thrust::get<2>(t);	
     	s[3][i] = s[0][i] + s[1][i] * s[2][i];	
    }
};



int main(void)
{
    // allocate storage
    thrust::device_vector<float> A(5);
    thrust::device_vector<float> B(5);
    thrust::device_vector<float> C(5);
    thrust::device_vector<float> D(5);

    // initialize input vectors
    A[0] = 3;  B[0] = 6;  C[0] = 2; 
    A[1] = 4;  B[1] = 7;  C[1] = 5; 
    A[2] = 0;  B[2] = 2;  C[2] = 7; 
    A[3] = 8;  B[3] = 1;  C[3] = 4; 
    A[4] = 2;  B[4] = 8;  C[4] = 3; 

    float *Aptr=thrust::raw_pointer_cast(&A[0]);
    float *Bptr=thrust::raw_pointer_cast(&B[0]);
    float *Cptr=thrust::raw_pointer_cast(&C[0]);
    float *Dptr=thrust::raw_pointer_cast(&D[0]);

    // apply the transformation
    thrust::for_each
    (
		     thrust::make_counting_iterator(0),
		     thrust::make_counting_iterator(5),
                     arbitrary_functor(Aptr,Bptr,Cptr,Dptr)
    );

    // print the output
    for(int i = 0; i < 5; i++)
        std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D[i] << std::endl;
}
