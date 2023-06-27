#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>

using namespace thrust::placeholders;

int main()
{
  thrust::device_vector<float> x(4);
  x[0] = 2;
  x[1] = 4;
  x[2] = 3;
  x[3] = 1;


  // vamos a normalizar un vector por su norma

  #ifdef INEFICIENTE
  //forma 1: 
  thrust::device_vector<float> xx(4);
  thrust::transform(x.begin(), x.end(),xx.begin(),_1*_1);
  float norma = sqrt(thrust::reduce(xx.begin(),xx.end()));	
  thrust::transform(x.begin(), x.end(), x.begin(),_1/norma);
  #else 	
  //forma 2:
  float norma = sqrt(thrust::transform_reduce(x.begin(), x.end(),_1*_1,0,thrust::plus<float>())); 
  thrust::transform(x.begin(), x.end(), x.begin(),_1/norma);
  #endif	

  for(int i=0;i<4;i++)
  std::cout << x[i] << std::endl;

  // deberia dar esto:
  // for i in 2. 4. 3. 1.; do echo "$i/sqrt(2*2+4*4+3*3+1*1)" | bc -l; done
  /* 	.36514837167011074230
	.73029674334022148461
	.54772255750516611345
	.18257418583505537115
  */
}






