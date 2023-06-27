#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
// note: functor inherits from unary_function
struct seno: public thrust::unary_function<float,float>
{
  __host__ __device__
  float operator()(float x) const
  {
    return sinf(x);
  }
};


int main(void)
{
  thrust::device_vector<float> v(4);
  v[0] = 1.0f; v[1] = 4.0f; v[2] = 9.0f; v[3] = 16.0f;

  // solo si c++11 soportado
  //auto first= thrust::make_transform_iterator(v.begin(), seno());

  //sino...
  typedef thrust::device_vector<float>::iterator FloatIterator;
  thrust::transform_iterator<seno, FloatIterator> first(v.begin(), seno());

  float result = thrust::reduce(first,first+4);

  float sum=0.0; for(int i=0;i<4;i++) sum+=sin(v[i]);
  std::cout << result << " vs " << sum << std::endl;

}
