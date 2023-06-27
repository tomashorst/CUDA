#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <fstream>
#include "simple_timer.h"

__global__ void kernel5(float *v, float *o,int N)
{
	extern __shared__ float sdata[];

   	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
   	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	// indice lineal de hilo en bloque, tid=0,1,...,blockDim.x*blockDim.y
	unsigned int tid = threadIdx.x + blockDim.x*threadIdx.y;

	// carga de los blockDim.x*blockDim.y productos 
	if(ix<N && iy<N)
	sdata[tid]=v[ix]*v[iy];
	else sdata[tid]=0.0;
	__syncthreads();

	// reduccion en arbol 
	for(unsigned int s=blockDim.x*blockDim.y/2; s>0; s>>=1) 
   	{
        	if (tid < s)
                sdata[tid] += sdata[tid + s];
        }
        __syncthreads();

	// el thread 0 del bloque guarda la suma parcial 
	unsigned int bid = blockIdx.x+gridDim.x*blockIdx.y;
	if(tid==0) o[bid]=sdata[0];
}

__global__ void kernel6(float *v, float *o,int N)
{
	extern __shared__ float sdata[];

   	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
   	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	// indice lineal de hilo en bloque, tid=0,1,...,blockDim.x*blockDim.y
	unsigned int tid = threadIdx.x + blockDim.x*threadIdx.y;

	// carga de los blockDim.x*blockDim.y productos 
	if(ix<N && iy<N)
	sdata[tid]=v[ix]*v[iy];
	else sdata[tid]=0.0;
	__syncthreads();

	// reduccion en arbol 
	for(unsigned int s=blockDim.x*blockDim.y/2; s>0; s>>=1) 
   	{
        	if (tid < s)
                sdata[tid] += sdata[tid + s];
        }
        __syncthreads();

	if(tid==0) atomicAdd(o,sdata[0]);
}

int test1(int argc,char **argv)
{
	int N=1024;
	if(argc>1) N=atof(argv[1]);

	cpu_timer reloj;
	reloj.tic();

	thrust::device_vector<float> values(N,1.0/N);
	float *v_raw=thrust::raw_pointer_cast(&values[0]);

	// 1024 threads	por cada 2d block
	size_t nthreads=32;	
	size_t nblocks=(N+nthreads-1)/nthreads;

	// nthreads*nthreads*nblocks*nblocks >= N^2
	dim3 nt(nthreads,nthreads);
	dim3 nb(nblocks,nblocks);

	// la matriz v[i]v[j] es subdividida an nblocks*nblocks bloques
	thrust::device_vector<float> parciales(nblocks*nblocks);
	float *parciales_raw=thrust::raw_pointer_cast(&parciales[0]);

	// memoria para el bloque, de dimension blockDim.x*blockDim.y
	size_t smem = nthreads*nthreads*sizeof(float); 

	kernel5<<<nb,nt,smem>>>(v_raw,parciales_raw,N);
	cudaDeviceSynchronize();

	// sumamos sobre todas los tiles o submatrices 
	float acum= thrust::reduce(parciales.begin(),parciales.end()) ;

	std::cout << "\nfinal result =" << acum << std::endl; 
	std::cout << "N= " << N << " ms= " << reloj.tac() << "\n" << std::endl;

	return 0;
}

int test2(int argc,char **argv)
{
	int N=1024;
	if(argc>1) N=atof(argv[1]);

	cpu_timer reloj;
	reloj.tic();

	thrust::device_vector<float> values(N,1.0/N);
	float *v_raw=thrust::raw_pointer_cast(&values[0]);

	// 1024 threads	por cada 2d block
	size_t nthreads=32;	
	size_t nblocks=(N+nthreads-1)/nthreads;

	// nthreads*nthreads*nblocks*nblocks >= N^2
	dim3 nt(nthreads,nthreads);
	dim3 nb(nblocks,nblocks);

	// la matriz v[i]v[j] es subdividida an nblocks*nblocks bloques
	thrust::device_vector<float> parciales(1);
	float *parciales_raw=thrust::raw_pointer_cast(&parciales[0]);

	// memoria para el bloque, de dimension blockDim.x*blockDim.y
	size_t smem = nthreads*nthreads*sizeof(float); 

	kernel6<<<nb,nt,smem>>>(v_raw,parciales_raw,N);
	cudaDeviceSynchronize();

	// sumamos sobre todas los tiles o submatrices 
	float acum= thrust::reduce(parciales.begin(),parciales.end()) ;

	std::cout << "\nfinal result =" << acum << std::endl; 
	std::cout << "N= " << N << " ms= " << reloj.tac() << "\n" << std::endl;

	return 0;
}

int main(int argc, char **argv){
	//test1(argc,argv);
	test2(argc,argv);
}
