#include <thrust/device_vector.h>
#include <thrust/transform.h>
using namespace thrust::placeholders;

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

  thrust::transform(x.begin(), x.end(), y.begin(), y.begin(),
    _1 + _2
  );
  // y es ahora {5, 5, 5, 5} --> HANDS-ON: comprobar!
  for(int i=0;i<4;i++)
  std::cout << y[i] << std::endl; //o usar printf...

}



