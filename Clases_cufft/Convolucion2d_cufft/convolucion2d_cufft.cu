// para compilar:
// nvcc -arch=sm_20 -o conv2d convolucion2d_cufft.cu -lcufft -DLAPLACIAN -DPRINTCONVOLUTION

#include<thrust/transform.h>
#include<thrust/for_each.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <fstream>
#include<cmath>

#include <cstdio>

// CUFFT include http://docs.nvidia.com/cuda/cufft/index.html
#include <cufft.h>
// Incluye ademas cufftComplex

/* counter-based random numbers */
// http://www.thesalmons.org/john/random123/releases/1.06/docs/
#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG

#define NX 76
#define NY 64
#define NRANK 2 
#define BATCH 1 

// Functor de Convolucion en espacion de Fourier
struct Convolutor
{
	__device__
  	cufftComplex operator()(cufftComplex c, unsigned int tid)
  	{
		// los indices bidimensionales de la transformada
		int ky=tid%(NY/2+1);
        	int kx=tid/(NY/2+1);

		// se corresponden a estos vectores de onda
		float qy=2.0*M_PI*ky/NY;
		float qx=(2.0*M_PI/NX)*((kx<NX/2)?(kx):(NX-kx));

		cufftReal fac;

		// fac=transformada del operador discretizado
		#ifdef LAPLACIAN
		fac=-(2.0-2.0*cosf(qx))-(2.0-2.0*cosf(qy)); // FFT del laplaciano discreto
		#elif defined(FILTER)
		//construir aqui su filtro en el espacio de Fourier, filtro(qx,qy)
		//fac= ...
		#else
		fac=1.0; // FFT de la identidad
		#endif

		cufftComplex res;

		// Como FFTinv(FFT)=NX*NY, normalizamos
		res.x = c.x*fac/(NX*NY);
		res.y = c.y*fac/(NX*NY);

		return res;
	}
};

struct fill_random_array
{
	unsigned long semi;
	fill_random_array(unsigned long semi_):semi(semi_)
	{};

	__device__
  	float operator()(unsigned int tid)
  	{
		int x=tid%(NY);
        	int y=int(tid*1.0/NY);

  		// keys and counters 
 		RNG philox;
    		RNG::ctr_type c={{}};
    		RNG::key_type k={{}};
    		RNG::ctr_type r;

  		// Garantiza una secuencia random "unica" para cada thread  
    		k[0]=semi;
    		c[1]=x;
    		c[0]=y;

      		r = philox(c, k);

		// gaussian random amplitudes
		float u1 = u01_open_closed_32_53(r[0]);
                float u2 = u01_open_closed_32_53(r[1]);
                float amp=sqrtf( -2.0*logf(u1) )*sinf(2.0*M_PI*u2);

		return amp;
  	}
};

struct fill_cosine_array
{
	__device__
  	float operator()(unsigned int tid)
  	{
		int y=tid%(NY);
        	int x=int(tid*1.0/NY);

		cufftReal facx=5.0;
		cufftReal facy=3.0;

		//return cosf(2.0*M_PI*x*facx/NX)*cosf(2.0*M_PI*y*facy/NY);
		return cosf(2.0*M_PI*x*facx/NX + 2.0*M_PI*y*facy/NY);
  	}
};



void llenar_array(thrust::device_vector<float> &d, unsigned long semi)
{
	#ifdef RANDOM
	// llena el array 2d en el device con numeros random
	thrust::transform(
		thrust::make_counting_iterator(0),thrust::make_counting_iterator(NX*NY),
		d.begin(),fill_random_array(semi)
	);
	#else
	// llena el array 2d en el device con una funcion bidimensional
	thrust::transform(
		thrust::make_counting_iterator(0),thrust::make_counting_iterator(NX*NY),
		d.begin(),fill_cosine_array()
	);
	#endif
}


int main()
{
	int semi=123456; // semilla global

	// Si i=0,...,NX-1; j=0,...,NY-1 --> NX*NY reales
	// Vamos a acceder asi en el espacio real: A(x,y)==A[y+NY*x]

	// Si FA=FFT(A), kx=0,...,NX; ky=0,...,NY/2+1 ---> NX*(NY/2+1) complejos
	// En el espacio de Fourier: FA(kx,ky)==FA[kx*(NY/2+1)+ky]

	unsigned long Nreals=NX*NY; // espacio real NX * NY floats
	unsigned long Ncomplex=NX*(NY/2+1); // espacio complejo, NX*NY+NX floats

	thrust::device_vector<cufftReal> input(Nreals); // array a transformar
	thrust::device_vector<cufftComplex> output(Ncomplex);// array transformado

	// llena el array 2d en el device con numeros random
	llenar_array(input,semi);

	// copia el array relleno al host
	thrust::host_vector<float> h_input_check(input);

	// creamos dos planes, uno para la transformada forward, otra para la backward
	cufftHandle plan, planinverso;
	int n[NRANK] = {NX, NY};

	// Crea plan 2D FFT-forward.
	if (cufftPlanMany(&plan, NRANK, n, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C,BATCH) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT Error: Unable to create plan r2c\n");
		return 1;
	}
	// ejecuta plan-forward
	if (cufftExecR2C(plan, thrust::raw_pointer_cast(&input[0]), thrust::raw_pointer_cast(&output[0])) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		return 1;
	}

	// multiplica el output=Fourier(input) por la transformada de algun "propagador",
	// "funcion de Green" u "operador" complejo para obtener la transformada de la convolucion
	thrust::transform(
		output.begin(),output.end(),
		thrust::make_counting_iterator(0),//construimos la transformada del Operador directamente en el functor
		output.begin(),
		Convolutor()
	);

	// Crea plan 2D FFT-backward.
	//int m[NRANK] = {NX, NY};
	if (cufftPlanMany(&planinverso, NRANK, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R,BATCH) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT Error: Unable to create plan c2r\n");
		return 1;
	}

	// ejecuta plan-backward, para obtener la convolucion en espacio real
	if (	cufftExecC2R(planinverso,
		thrust::raw_pointer_cast(&output[0]),
		thrust::raw_pointer_cast(&input[0])) != CUFFT_SUCCESS	)
	{
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		return 1;
	}

	// copia la convolucion en espacio real invFourier(Fourier(input)Fourier(Operador)) al host
	thrust::host_vector<float> h_input(input);

	// checking
	float error, errormax;
	for(int i=0;i<NX;i++){
	for(int j=0;j<NY;j++){
			#ifdef LAPLACIAN // si Operador=Laplaciano 
			float lap=
			h_input_check[(j-1+NY)%NY+NY*i]+h_input_check[(j+1)%NY+NY*i]+
			h_input_check[j+NY*((i+1)%NX)]+h_input_check[j+NY*((i-1+NX)%NX)]-
			4.0*h_input_check[j+NY*i];
			error=fabs(h_input[j+NY*i]-lap);
			#else // Operador = Identidad
			error=fabs(h_input[j+NY*i]-h_input_check[j+NY*i]);
			#endif
			errormax = (error>errormax)?(error):(errormax);
		}
	}
	std::cout << "max error=" << errormax << std::endl;

	#ifdef PRINTCONVOLUTION
	std::ofstream convolout("convolution.dat");
	std::ofstream inputout("input.dat");
	for(int i=0;i<NX;i++){
		for(int j=0;j<NY;j++){
			inputout << h_input_check[j+NY*i] << " ";
			convolout << h_input[j+NY*i] << " ";
		}
		convolout << "\n";
		inputout << "\n";
	}

	// por ejemplo, en el caso del laplaciano, chequear:

	// Si el input es cos(2*pi*x/64)*cos(y*pi*2/64) con x=0,...,NX e y=0,...,NY
	// gnuplot> splot "input.dat" matrix w pm3d, cos(2*pi*x/NX)*cos(y*pi*2/NY)

	// El laplaciano de cos(2*pi*x/NX)*cos(y*pi*2/NY) es -[(2*pi/NX)**2+(2*pi/NY)**2]*cos(2*pi*x/NX)*cos(y*pi*2/NY)
	// gnuplot> splot "convolution.dat" matrix w pm3d, -((2*pi/NX)**2+(2*pi/NY)**2)*cos(2*pi*x/NX)*cos(y*pi*2/NY)
	#endif

	// Imprime el modulo cuadrado de la transformada de Fourier
	#ifdef PRINTSTRUCTUREFACTOR
	std::ofstream sofqout("strucinput.dat");
	thrust::host_vector<cufftComplex> h_output(output);
	for(int i=0;i<NX;i++){
	for(int j=0;j<NY/2+1;j++){
		cufftComplex C = h_output[j+(NY/2+1)*i]; // check size=(NY/2+1)+(NY/2+1)(NX-1)=(NY/2+1)*NX
		sofqout << C.x*C.x+C.y*C.y << " ";
	}
	sofqout << "\n";
	}
	#endif
}



