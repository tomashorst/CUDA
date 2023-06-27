#include "curso.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <iostream>

struct mi_operacion // normaliza por N
{
      public:	
      float Norm;
      mi_operacion(float suma){Norm=suma;};
	  __device__
	  float operator()(float a)
	  {
		  return a/Norm;
	  }
};


int main()
{
  thrust::device_vector<float> x(4), y(4);
  x[0] = 2;
  x[1] = 4;
  x[2] = 3;
  x[3] = 1;

  float suma = thrust::reduce(x.begin(), x.end());
  thrust::transform(x.begin(), x.end(), x.begin(),mi_operacion(suma));
  // HANDS-ON: chequear que este normalizado...

  for(int i=0;i<4;i++)
  std::cout << x[i] << std::endl;

}





