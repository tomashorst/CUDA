/*
Resuelve el modelo de Ising con el metodo de metropolis MonteCarlo. 
Es decir, genera una cadena de Markov que samplea configuraciones 
{S_1,...,S_N} con energias

H = - \sum_{<ij>} S_i S_j

donde \sum_<ij> denota suma sobre pares de primetos vecinos,
y S_i=+-1 el "spin". El metodo de MonteCarlo garantiza 
que para un numero de "updates" lo suficientemente grande, 
las configuraciones son sampleadas segun su peso de Boltzmann 

P~exp[-H/T]

permitiendo hacer promedios termodinamicos de equilibrio.
En particular, la magnetizacion por sitio en funcion de la temperatura

M = \sum_i S_i/N

con N=L*L el numero total de sitios en una 
red cuadrada de LxL.

La estrategia de paralelizacion es la del tablero de ajedrez:
hacemo un updates paralelo de todas las fichas blancas, seguido 
por uno de todas las negras. 

Se puede demostrar que la cadena de Markov converge a equilibrio.
*/

#include<iostream> //input-output pantalla
#include<fstream> //input-output disco
#include<cstdlib> // atoi, atof, etc
#include <unistd.h> // getop
#include <cmath> 

#include "gpu_timer.h"
#include "cpu_timer.h"

// thrust
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include<thrust/iterator/counting_iterator.h>

/* counter-based random numbers */
// http://www.thesalmons.org/john/random123/releases/1.06/docs/
#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG


// functor para obtener numeros aleatorios uniformes en [0,1]
// a partir de un key (sitio), un global seed, y un tiempo 
__device__
float uniform(int n, int seed, int t)
{
		// keys and counters 
		RNG philox; 	
		RNG::ctr_type c={{}};
		RNG::key_type k={{}};
		RNG::ctr_type r;
		// Garantiza una secuencia random "unica" para cada thread	
		k[0]=n;    // distintos sitios, distintos numeros random!
		c[1]=seed; // seed global, necesario para decidir reproducir secuencia, o no...
		c[0]=t;    // el seed tiene que cambiar con la iteracion, sino...
		r = philox(c, k); // son dos numeros random, usaremos uno solo r[0]
		return (u01_closed_closed_32_53(r[0])); // funcion adaptadora a [0,1]
}


// functor tipo predicado: cada thread determina si su sitio es del "color" 0 o 1
// necesario para "transform_if"
struct ficha
{
	bool color;
	int L;
	ficha(bool _color, int _L):color(_color),L(_L){};

	__device__ __host__
	bool operator()(int n){
		return ((n%L+int(n/L))%2==color); // true si n es color
	}	
};

// functor: cada thread hace el metropolis update de su sitio
// ejemplo para modelo de Ising con interaccion de primeros vecinos
struct metropolis
{
	int L;
	float T;
	int *Mptr;
	int t;
	int seed;
	metropolis(int * _Mptr, float _T, int _L, int _t, int _seed):
	Mptr(_Mptr),T(_T),L(_L),t(_t),seed(_seed){};	

	__device__
	int operator()(int n){

		int nx=n%L;
	       	int ny=int(n/L);

		int local_field=
		Mptr[(nx-1+L)%L  + ny*L] + Mptr[(nx+1+L)%L  + ny*L] +
		Mptr[nx+((ny+1+L)%L)*L]  + Mptr[nx+((ny-1+L)%L)*L];

		// contribucion de nuestro spin sin flipear a la energia  
		float ene0=-Mptr[n]*local_field;	

		// contribucion a la energia de nuestro spin flipeado
		//int ene1=M[n]*vecinos;
		float ene1=Mptr[n]*local_field;	

		// metropolis: aceptar flipeo solo si r < exp(-(ene1-ene0)/temp)
		float p=exp(-(ene1-ene0)/T);

		// numero random entre [0,1] uniforme
		//float r=float(rand())/RAND_MAX;->philox
		float rn = uniform(n, seed, t);

		// metropolis update segun regla de acceptancia 
		return (rn<p)?(-Mptr[n]):(Mptr[n]);
	}	
};

// imprime toda la red de sitios en pantalla
void print_campo_de_magnetizacion(thrust::device_vector<int> &M, int L, std::ofstream &fout)
{
	fout << "\n";
	for(int ny=0;ny<L;ny++){
		for(int nx=0;nx<L;nx++){
			fout << M[nx+ny*L] << "\t"; 
		}
		fout << std::endl;
	}
}

/*
COMPILACION:
nvcc miniising.cu
*/
int main(int argc, char **argv)
{
	int globalseed=123456; // semilla global generador paralelo
	int L=512; // largo red cuadrada
	float T=1.0; // temperatura
	int nrun=1000; // numero total de pasos de MonteCarlo
	int tsnap=nrun+1; // cada cuando imprimo configs
	std::ofstream fout("movie.dat"); // fichero para guardar configs
	std::ofstream mout("mag.dat"); // fichero para guardar magnetizacion por sitio vs tiempo

	// para tomar opciones de la linea de comandos
	int opt;
	while ((opt = getopt(argc, argv, "l:r:T:s:w:")) != -1) 
	{
               switch (opt) {
               case 'l':
                   L = atoi(optarg);
                   break;
               case 'r':
                   nrun = atoi(optarg);
                   break;
               case 'T':
                   T = atof(optarg);
                   break;
               case 's':
                   globalseed = atoi(optarg);
                   break;
               case 'w':
                   tsnap = atoi(optarg);
                   break;
               default: /* '?' */
                   fprintf(stderr, "Uso: %s [-l L] [-r niter] [-T temp] [-s semilla] [-w snap]\n", argv[0]);
                   exit(EXIT_FAILURE);
               }
	}

	std::ofstream logout("log.dat"); // fichero para guardar configs
	logout << "L=" << L << ", ";
	logout << "nrun=" << nrun << ", ";
	logout << "T=" << T << ", ";
	logout << "globalseed= " << globalseed << " ";
	logout << "tsnap= " << tsnap << "\n";	
	std::cout << "L=" << L << ", ";
	std::cout << "nrun=" << nrun << ", ";
	std::cout << "T=" << T << ", ";
	std::cout << "globalseed= " << globalseed << " ";
	std::cout << "tsnap= " << tsnap << "\n";	
	#ifdef _OPENMP
  	logout << "#host OMP threads = " << omp_get_max_threads() << std::endl;
  	std::cout << "#host OMP threads = " << omp_get_max_threads() << std::endl;
  	#elif defined(__CUDACC__)
  	int card;
  	cudaGetDevice(&card);
  	cudaDeviceProp deviceProp;
  	cudaGetDeviceProperties(&deviceProp, card);
  	logout << "\nDevice Selected " << card << " " << deviceProp.name << "\n";
	std::cout << "\nDevice Selected " << card << " " << deviceProp.name << "\n";
	#else
	std::cout << "\nCPU serial\n";
	logout << "\nCPU serial\n";
  	#endif

	/**************************************************/
	// AHORA EMPIEZA LO IMPORTANTE


	if(L%2==1) std::cout << "warning: para usar checkerboard L debe ser par" << std::endl;

	int N=L*L;

	// vector de magnetizacion en la red
	thrust::device_vector<int> M(N);
	int *Mraw=thrust::raw_pointer_cast(M.data()); // puntero crudo

	// condicion inicial random, usando generador standard de C
	for(int n=0;n<N;n++) M[n]=(rand()*1.0/RAND_MAX>0.5)?(1):(-1);


	cpu_timer crono;
	crono.tic();

	// loop de pasos de MonteCarlo
	for(int nt=0;nt<nrun;nt++)
	{
		//ojo!: lento, GPU->CPU->disco de mucha info! (hacer cada muchos pasos)
		if((nt+1)%tsnap==0) print_campo_de_magnetizacion(M, L, fout);

		// imprime magnetizacion por sitio (usando parallel reduction)
		mout << thrust::reduce(M.begin(),M.end())*1.0/N << std::endl;

		for(int color=0;color<2;color++){ // "checkerboard decomposition"
			// update de sitios de color "color" usando transform_if paralelo
			thrust::transform_if(
				thrust::make_counting_iterator(0), thrust::make_counting_iterator(N), //rango
				M.begin(), // output
				metropolis(Mraw,T,L,nt,globalseed), // operacion 
				ficha(color, L) // predicado
			);
		}
	}

	std::cout << "ejecutable = " << argv[0] << std::endl;
	std::cout << "ms = " << crono.tac() << std::endl;


	logout << "ejecutable = " << argv[0] << std::endl;
	logout << "ms = " << crono.tac() << std::endl;


	return 0;
}
