#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include "simple_timer.h"

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cassert>

#define USE_NVTX
#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif


// This example demonstrates how to intercept calls to get_temporary_buffer
// and return_temporary_buffer to control how Thrust allocates temporary storage
// during algorithms such as thrust::sort. The idea will be to create a simple
// cache of allocations to search when temporary storage is requested. If a hit
// is found in the cache, we quickly return the cached allocation instead of
// resorting to the more expensive thrust::cuda::malloc.
//
// Note: this implementation cached_allocator is not thread-safe. If multiple
// (host) threads use the same cached_allocator then they should gain exclusive
// access to the allocator before accessing its methods.

struct not_my_pointer
{
  not_my_pointer(void* p)
    : message()
  {
    std::stringstream s;
    s << "Pointer `" << p << "` was not allocated by this allocator.";
    message = s.str();
  }

  virtual ~not_my_pointer() {}

  virtual const char* what() const
  {
    return message.c_str();
  }

private:
  std::string message;
};

// A simple allocator for caching cudaMalloc allocations.
struct cached_allocator
{
  typedef char value_type;

  cached_allocator() {}

  ~cached_allocator()
  {
    free_all();
  }

  char *allocate(std::ptrdiff_t num_bytes)
  {
    std::cout << "cached_allocator::allocate(): num_bytes == "
              << num_bytes
              << std::endl;

    char *result = 0;

    // Search the cache for a free block.
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if (free_block != free_blocks.end())
    {
      std::cout << "cached_allocator::allocate(): found a free block"
                << std::endl;

      result = free_block->second;

      // Erase from the `free_blocks` map.
      free_blocks.erase(free_block);
    }
    else
    {
      // No allocation of the right size exists, so create a new one with
      // `thrust::cuda::malloc`.
      try
      {
        std::cout << "cached_allocator::allocate(): allocating new block"
                  << std::endl;

        // Allocate memory and convert the resulting `thrust::cuda::pointer` to
        // a raw pointer.
        result = thrust::cuda::malloc<char>(num_bytes).get();
      }
      catch (std::runtime_error&)
      {
        throw;
      }
    }

    // Insert the allocated pointer into the `allocated_blocks` map.
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return result;
  }

  void deallocate(char *ptr, size_t)
  {
    std::cout << "cached_allocator::deallocate(): ptr == "
              << reinterpret_cast<void*>(ptr) << std::endl;

    // Erase the allocated block from the allocated blocks map.
    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);

    if (iter == allocated_blocks.end())
      throw not_my_pointer(reinterpret_cast<void*>(ptr));

    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // Insert the block into the free blocks map.
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

private:
  typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
  typedef std::map<char*, std::ptrdiff_t>      allocated_blocks_type;

  free_blocks_type      free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all()
  {
    std::cout << "cached_allocator::free_all()" << std::endl;

    // Deallocate all outstanding blocks in both lists.
    for ( free_blocks_type::iterator i = free_blocks.begin()
        ; i != free_blocks.end()
        ; ++i)
    {
      // Transform the pointer to cuda::pointer before calling cuda::free.
      thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
    }

    for( allocated_blocks_type::iterator i = allocated_blocks.begin()
       ; i != allocated_blocks.end()
       ; ++i)
    {
      // Transform the pointer to cuda::pointer before calling cuda::free.
      thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
    }
  }
};

int N;
#define NSTREAMS	3

std::ofstream logout("log.dat");

void test1()
{
	thrust::device_vector<float> output(N);
	thrust::device_vector<float> values(N,(1.0/N));

        cached_allocator alloc;

	cpu_timer reloj;
	reloj.tic();

	auto irrelevante=thrust::make_discard_iterator();

	cudaStream_t s[NSTREAMS];
  	for(int i=0;i<NSTREAMS;i++) cudaStreamCreate(&s[i]);

	int n;

	for(int i=0;i<N;i++){

		auto keys=thrust::make_constant_iterator(i);
		thrust::reduce_by_key(thrust::cuda::par(alloc).on(s[n]),
			keys,keys+N,values.begin(),
			irrelevante,
			output.begin()+i
		);
		n++;
		n=n%NSTREAMS;		
	}

  	for(int i=0;i<NSTREAMS;i++) cudaStreamSynchronize(s[i]);

	// debug check de sumas parciales
	//for(int i=0;i<N;i++)
	//std::cout << output[i] << std::endl;

	logout << "final result 1=" << thrust::reduce(output.begin(),output.end()) << std::endl; 
	logout << "En ms=" << reloj.tac() << std::endl;
}


void test2()
{
	thrust::cuda::vector<float> values(N,(1.0/N));

	cpu_timer reloj;
	reloj.tic();

        cached_allocator alloc;

	float acum=0.0;
	for(int i=0;i<N;i++){
		acum+=thrust::reduce(thrust::cuda::par(alloc),values.begin(),values.end());
	}

	logout << "final result 2=" << acum << std::endl; 
	logout << "En ms=" << reloj.tac() << std::endl;
}

void test3()
{
	thrust::cuda::vector<float> values(N,(1.0/N));

	cpu_timer reloj;
	reloj.tic();

	float acum=0.0;
	for(int i=0;i<N;i++){
		acum+=thrust::reduce(values.begin(),values.end());
	}

	logout << "final result 3=" << acum << std::endl; 
	logout << "En ms=" << reloj.tac() << std::endl;
}


__global__ void kernel4(float *v, float *o,int N)
{
	extern __shared__ float sdata[];

   	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
   	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	// indice lineal de hilo en bloque, tid=0,1,...,blockDim.x*blockDim.y
	unsigned int tid = threadIdx.x + blockDim.x*threadIdx.y;

	if(ix<N && iy<N)
	sdata[tid]=v[ix]*v[iy];
	else sdata[tid]=0.0;
	
	__syncthreads();

	// reduccion serial...
	if(tid==0){
		float acum=0.0;	

		// indice linea de bloque en la grilla
		unsigned int bid = blockIdx.x+gridDim.x*blockIdx.y;
		for(int i=0;i<blockDim.x*blockDim.y;i++) acum += sdata[i];
		o[bid]=acum;	
	}
}

void test4()
{
	cpu_timer reloj;
	reloj.tic();

	thrust::device_vector<float> values(N,1.0/N);
	float *v_raw=thrust::raw_pointer_cast(&values[0]);

	size_t nthreads=32;
	size_t nblocks=(N+nthreads-1)/nthreads;

	// 1 block 1024 threads	

	// nthreads*nthreads*nblocks*nblocks >= N^2
	dim3 nt(nthreads,nthreads);
	dim3 nb(nblocks,nblocks);

	thrust::device_vector<float> parciales(nblocks*nblocks);
	float *parciales_raw=thrust::raw_pointer_cast(&parciales[0]);

	size_t smem = nthreads*nthreads*sizeof(float); 

	kernel4<<<nb,nt,smem>>>(v_raw,parciales_raw,N);
	cudaDeviceSynchronize();

	float acum= thrust::reduce(parciales.begin(),parciales.end()) ;
	logout << "final result 4=" << acum << std::endl; 
	logout << "En ms=" << reloj.tac() << std::endl;

	for(int i=0;i<nblocks*nblocks;i++)
	std::cout << parciales[i] << " ";
	std::cout << std::endl;

}

void test5()
{
	cpu_timer reloj;
	reloj.tic();

	thrust::device_vector<float> values(N,1.0/N);

	float acum=0.0;
	for(int i=0;i<N;i++){
		float value0=values[i];
		auto const_it=thrust::make_constant_iterator(value0);
		acum+=thrust::inner_product(values.begin(),values.end(),const_it,0.0);
	}

	logout << "final result 5=" << acum << std::endl; 
	logout << "En ms=" << reloj.tac() << std::endl;	
}


int main(int argc, char **argv){

	if(argc>1) N = atoi(argv[1]);
	else N=128;

	PUSH_RANGE("Test1",1)	
	test1();
	POP_RANGE

	PUSH_RANGE("Test2",2)	
	test2();
	POP_RANGE

	PUSH_RANGE("Test3",3)	
	test3();
	POP_RANGE

	PUSH_RANGE("Test4",4)	
	test4();
	POP_RANGE

	PUSH_RANGE("Test4",4)	
	test5();
	POP_RANGE

/*	float acum=0.0;
	for(int i=0;i<N;i++){
	for(int j=0;j<N;j++){
		acum+=(1.0/N)*(1.0/N);
	}}
	std::cout << "check=" << acum << std::endl;
*/
	return 0;
}

