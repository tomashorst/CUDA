#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <curand_kernel.h>

#include <iostream>
#include <iomanip>

#define GLOBALSEED  12345

// we could vary M & N to find the perf sweet spot

struct estimate_pi : 
    public thrust::unary_function<unsigned int, float>
{
  __device__
  float operator()(unsigned int thread_id)
  {
    float sum = 0;
    unsigned int N = 10000; // samples per thread
    unsigned int seed = thread_id;

    curandStatePhilox4_32_10_t s;

    // seed a random number generator
    curand_init(GLOBALSEED,seed, 0, &s);

    // take N samples in a quarter circle
    for(unsigned int i = 0; i < N; ++i)
    {
      // draw a sample from the unit square
      float x = curand_uniform(&s);
      float y = curand_uniform(&s);

      // measure distance from the origin
      float dist2 = (x*x + y*y);

      // add 1.0f if (u0,u1) is inside the quarter circle
      if(dist2 <= 1.0f)
        sum += 1.0f;
    }

    // multiply by 4/N to get the area of the whole circle
    sum *= 4.0f/N;

    return sum;
  }
};

int main(void)
{
  // use 30K independent seeds
  int M = 30000;

  float estimate = thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(M),
        estimate_pi(),
        0.0f,
        thrust::plus<float>());
  estimate /= M;

  std::cout << std::setprecision(6);
  std::cout << "pi is approximately ";
  std::cout << estimate << std::endl;
  return 0;
}

//if(thread_id<=2) printf("%d %d %d\n",thread_id,s.key.x,s.ctr.x);
//if(thread_id<=2) printf("%d %d %d\n",thread_id,s.key.x,s.ctr.x);
