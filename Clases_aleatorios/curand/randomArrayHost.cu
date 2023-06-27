/*
  Ejercicios: 
  Timming CPU vs GPU.
  Cambiar y Verificar distribucion.      
*/

/*
 * This program uses the host CURAND API to generate 100 
 * pseudorandom floats.
 */
 #include <stdio.h>
 #include <stdlib.h>
 #include <cuda.h>
 #include <curand.h>
 
 #define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
     printf("Error at %s:%d\n",__FILE__,__LINE__);\
     return EXIT_FAILURE;}} while(0)
 #define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
     printf("Error at %s:%d\n",__FILE__,__LINE__);\
     return EXIT_FAILURE;}} while(0)
 
 int main(int argc, char *argv[])
 {

     size_t n = 100;

     if(argc>1) n=atoi(argv[1]);

     size_t i;
     curandGenerator_t gen;
     float *hostData;
 
     /* Allocate n floats on host */
     hostData = (float *)calloc(n, sizeof(float));
 
    #ifndef CPU
    float *devData; 

     /* Allocate n floats on device */
     CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(float)));
 
     /* Create pseudo-random number generator */
     CURAND_CALL(curandCreateGenerator(&gen, 
                 CURAND_RNG_PSEUDO_DEFAULT));
     
     /* Set seed */
     CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                 1234ULL));
 
     /* Generate n floats on device */
     CURAND_CALL(curandGenerateUniform(gen, devData, n));
 
     /* Copy device memory to host */
     CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(float),
         cudaMemcpyDeviceToHost));
    #else    

     /* Create pseudo-random number generator */
     CURAND_CALL(curandCreateGeneratorHost(&gen, 
        CURAND_RNG_PSEUDO_DEFAULT));

     /* Set seed */
     CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
        1234ULL));

     /* Generate n floats on device */
     CURAND_CALL(curandGenerateUniform(gen, hostData, n));

     #endif

     /* Show result */
     int m=(n>100)?100:n;
     for(i = 0; i < m; i++) {
         printf("%1.4f ", hostData[i]);
     }
     printf("\n");
 
     /* Cleanup */
     CURAND_CALL(curandDestroyGenerator(gen));
 
     #ifndef CPU
     CUDA_CALL(cudaFree(devData));
     #endif
     free(hostData);    

     return EXIT_SUCCESS;
 }
 