#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <iostream>
#include <iomanip>
#include <cmath>

/* counter-based random numbers */
// http://www.thesalmons.org/john/random123/releases/1.06/docs/
#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG

#include "cpu_timer.h"

struct estimate_pi
{

   unsigned long semi;
   unsigned int N; // muestras por thread
   estimate_pi(int _N, unsigned long _semi):N(_N),semi(_semi){};

  __device__
  double operator()(unsigned int thread_id)
  {
    // keys and counters 
    RNG philox;
    RNG::ctr_type c={{}};
    RNG::key_type k={{}};
    RNG::ctr_type r;

    // Garantiza una secuencia random "unica" para cada thread	
    k[0]=thread_id; 
    c[1]=semi;

    float sum = 0;

    // cada thread tira N dardos 
    for(unsigned int i = 0; i < N; ++i)
    {
      c[0]=i; // garantiza una secuencia de N nros descorrelacionados para el thread "i"	

      // una llamada retorna dos numeros random descorrelacionados	
      r = philox(c, k); 

      // sampleamos un punto dentro del cuadrado unidad [0,1][0,1]
      double x=(u01_closed_closed_32_53(r[0]));
      double y=(u01_closed_closed_32_53(r[1]));
	
      // mido distancia al origen
      //float dist = sqrtf(x*x + y*y); // no necesito el sqrtf
      double dist = (x*x + y*y); // no necesito el sqrtf

      // sumo 1.0f si (u0,u1) esta dentro del circulo
      if(dist <= 1.0f)
        sum += 1.0f;
    }

    // multiplico por 4 para tener el area del circulo completo...
    sum *= 4.0f;

    // divido por el numero N de dardos tirados por este thread
    return sum / N;
  }
};

#include <omp.h>
using namespace thrust;	

int main(int argc, char **argv)
{

  #ifdef _OPENMP
  std::cout << "#host OMP threads = " << omp_get_max_threads() << std::endl;
  #elif defined(__CUDACC__)
  int card;
  cudaGetDevice(&card);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, card);
  std::cout << "\nDevice Selected " << card << " " << deviceProp.name << "\n";
  #endif			

  // uso 300K tiradores (semillas) independientes...
  int tiradores = 10000;
  if(argc>1) tiradores=atoi(argv[1]);

  // cada tirador tira 10K dardos...
  int dardosportirador = 10000;	
  if(argc>2) dardosportirador=atoi(argv[2]);

  // una semilla global para que cada corrida tenga otras secuencias aleatorias
  int semillaglobal = 0;	
  if(argc>3) semillaglobal=atoi(argv[3]);

  std::cout << "Ud esta por poner " << tiradores << " tiradores a tirar " << dardosportirador << " cada uno" << std::endl;
  std::cout << "O sea, va a tirar " << long(tiradores)*dardosportirador << " en total"<< std::endl;
  std::cout << "La semilla global es " << semillaglobal << std::endl;


  cpu_timer crono;
  crono.tic();	

  // "thrust::tranform_reduce(), fusiona dos operaciones [best practice]	 	
  // Parte "transform": Pongo 30K tiradores independientes a tirar en paralelo 
  // y a realizar sus estimaciones de pi. El functor estimate_pi() contiene 
  // las instrucciones (identicas) para cada tirador.	
  // Parte "reduce": sumo sus estimaciones de pi.
  double estimate;
  estimate = transform_reduce(
		counting_iterator<int>(0),
                counting_iterator<int>(tiradores),
                estimate_pi(dardosportirador,semillaglobal),
                double(0.0f),
                thrust::plus<double>()
	    );


  estimate /= tiradores; // 3*10^6 dardos tirados...(comparar tiempos con CPU)  

  std::cout << std::setprecision(7);
  std::cout << "\n==\n pi es aproximadamente " << estimate << "\n==\n" << std::endl;
  std::cout << argv[0] << ": " << crono.tac() << " ms" << std::endl;  

  return 0;
}


// GPU    --> nvcc monte_carlo_Pi.cu -I ../common/ -D__STDC_CONSTANT_MACROS -arch=sm_21
// OPENMP --> g++ -O2 monte_carlo_Pi.cpp -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -I /usr/local/cuda-5.0/include/ -I ../common/ -D__STDC_CONSTANT_MACROS


