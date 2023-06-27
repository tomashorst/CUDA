#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <cmath>
#include <iostream>

struct mi_operacion
{
	  __device__
	  float operator()(float a, float b)
	  {
		  return sqrt(a+b);
	  }
};

int main()
{
  thrust::device_vector<float> x(4), y(4);
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  x[3] = 4;

  y[0] = 4;
  y[1] = 3;
  y[2] = 2;
  y[3] = 1;

  thrust::transform(x.begin(), x.end(), y.begin(), y.begin(),mi_operacion());

  for(int i=0;i<4;i++)
  std::cout << y[i] << std::endl;
}




