#include <thrust/device_ptr.h>
#include <thrust/device_free.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

using namespace thrust::placeholders;
int main(void)
{
  float *raw_ptr;
  cudaMalloc((void **)&raw_ptr, 4*sizeof(float));
  thrust::device_ptr<float> x(raw_ptr);

  x[0] = 2;
  x[1] = 4;
  x[2] = 3;
  x[3] = 1;

  float suma = thrust::reduce(x, x+4);
  thrust::transform(x,x+4,x,_1/suma);

  for(int i=0;i<4;i++) std::cout << x[i] << std::endl;

  thrust::device_free(x);
  return 0;
}



