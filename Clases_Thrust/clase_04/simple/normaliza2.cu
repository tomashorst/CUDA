#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

using namespace thrust::placeholders;

int main()
{
  thrust::device_vector<float> x(4);
  x[0] = 2;
  x[1] = 4;
  x[2] = 3;
  x[3] = 1;

  float suma = thrust::reduce(x.begin(), x.end());
  thrust::transform(x.begin(), x.end(), x.begin(),_1/suma);
  // HANDS-ON: chequear que este normalizado...

  for(int i=0;i<4;i++)
  std::cout << x[i] << std::endl;
}






