#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <stdio.h>
using namespace thrust::placeholders;

int main()
{
  thrust::device_vector<float> x(4), y(4);
  x[0] = 2;
  x[1] = 4;
  x[2] = 6;
  x[3] = 8;

  y[0] = 4;
  y[1] = 3;
  y[2] = 2;
  y[3] = 1;
  float a=0.5;

  thrust::transform(x.begin(), x.end(), y.begin(), y.begin(),
    a*_1 + _2
  );
  // y es ahora {5, 5, 5, 5} --> HANDS-ON: comprobar!
  for(int i=0;i<4;i++)
  printf("%f\n", float(y[i]));
}





