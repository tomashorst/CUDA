/*
 * This program uses the host CURAND API to generate 100 
 * pseudorandom floats.
 */

 #include <iostream>
 #include <stdlib.h>
 #include <cuda.h>
 #include <curand.h>
 #include <thrust/device_vector.h>
 #include <thrust/count.h>

 #define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
     printf("Error at %s:%d\n",__FILE__,__LINE__);\
     return EXIT_FAILURE;}} while(0)
 #define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
     printf("Error at %s:%d\n",__FILE__,__LINE__);\
     return EXIT_FAILURE;}} while(0)
 

struct dentroCuadradoUnidad
{
    __host__ __device__
    bool operator()(thrust::tuple<float,float> tup)
    {
        float x=thrust::get<0>(tup);
        float y=thrust::get<1>(tup);
        return (x*x+y*y)<1;
    }
};

 int main(int argc, char *argv[])
 {
    size_t n = 10000000;
    curandGenerator_t gen;
 
    thrust::device_vector<float> devData(n);

     /* Create pseudo-random number generator */
     CURAND_CALL(curandCreateGenerator(&gen, 
                 CURAND_RNG_PSEUDO_DEFAULT));
     
     /* Set seed */
     CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                 1234ULL));
 
    /* Generate n floats on device */
    float *devData_raw=thrust::raw_pointer_cast(&devData[0]);
    CURAND_CALL(curandGenerateUniform(gen, devData_raw, n));
  
    // usamos la primera mitad como x, la segunda como y
    int adentro = thrust::count_if(
        thrust::make_zip_iterator((thrust::make_tuple(devData.begin(),devData.begin()+n/2))),
        thrust::make_zip_iterator((thrust::make_tuple(devData.begin()+n/2,devData.begin()+n))),
        dentroCuadradoUnidad()
    );    

    /*
    dardos adentro = rho pi R^2/4 
    dardos total = rho R^2
    4*adentro/total = pi
    */

    // hay n/2 dardos ->
    std::cout << 4.0*adentro*(2.0/n) << std::endl;
 
     /* Cleanup */
     CURAND_CALL(curandDestroyGenerator(gen));
     return EXIT_SUCCESS;
 }