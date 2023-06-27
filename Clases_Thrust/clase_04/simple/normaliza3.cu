#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

__global__ 
void kernel_normaliza(float *x, float suma, int dim)
{
    int id = threadIdx.x + (blockIdx.x * blockDim.x); 
    if (id < dim)
    {
         x[id] = x[id]/suma;
    }
}

int main()
{
  thrust::device_vector<float> x(4);

  x[0] = 2;
  x[1] = 4;
  x[2] = 3;
  x[3] = 1;

  float suma = thrust::reduce(x.begin(), x.end());

  float * x_ptr = thrust::raw_pointer_cast(&x[0]);

  kernel_normaliza<<<1,4>>>(x_ptr,suma,4); 
  // HANDS-ON: chequear que este normalizado...

  for(int i=0;i<4;i++)
  std::cout << x[i] << std::endl;
}






