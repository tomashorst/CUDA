/*
Este programita simplemente transforma Fourier una senial unidimensional, 
almacenada en CPU, usando fftw y thrust. El "device" es la misma CPU. 
*/

/* algunos headers de la libreria thrust */
// https://github.com/thrust/thrust/wiki/Documentation
#include<thrust/transform.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include "cpu_timer.h"
#include "openmp_timer.h"
#include <omp.h>
#include <complex>
#include<fftw3.h>


using namespace std;

/* + Array Size N: use only powers of 2 */
#ifndef TAMANIO
#define N 1048576
#else
#define N TAMANIO
#endif

typedef double REAL;
typedef complex<double> COMPLEX;

/* Parametros de la senial */
#define A1 4
#define A2 6
#define T1 N/4
#define T2 N/8
struct FillSignal
{
	__device__ __host__
	REAL operator()(unsigned tid)
    	{
		// ponga aqui su funcion preferida...
		return A1*2.0*cosf(2*M_PI*tid*T1/(float)N) + A2*2.0*sinf(2*M_PI*tid*T2/(float)N);
    	}
};


///////////////////////////////////////////////////////////////////////////
int main(void) {
	// Un container de thrust para guardar el input real en CPU 
	thrust::host_vector<REAL> D_input(N);

	// toma el raw_pointer del array de input, para pasarselo a FFTW luego
	REAL *d_input = thrust::raw_pointer_cast(&D_input[0]);

	// Un container de thrust para guardar el ouput complejo en CPU = transformada del input 
	int Ncomp = N/2+1;
	thrust::host_vector<COMPLEX> D_output(Ncomp);

	// toma el raw_pointer del array de output, para pasarselo a FFTW luego
	COMPLEX *d_output = thrust::raw_pointer_cast(&D_output[0]); 


	// crea el plan de transformada de FFTW
#ifdef FFTWTHREADS
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	cout << "#nro omp threads = " << omp_get_max_threads() << endl;
#endif
	fftw_plan  plan_d2z = fftw_plan_dft_r2c_1d(N, d_input, reinterpret_cast<fftw_complex*>(d_output), FFTW_ESTIMATE);

	// ---- Start ---- 
	//lleno el array de tamanio N con la senial, a travez del functor "FillSignal"
	thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(N),D_input.begin(),FillSignal());

	// un timer para CPU
#ifdef FFTWTHREADS
	omp_timer t;
#else
	cpu_timer t;
#endif
	t.tic();

	//Transforma Fourier ejecutando el plan
	fftw_execute(plan_d2z);

	t.tac();
	// ---- Stop ---- 

	// Imprime la transformada 
	cout << "# Tamanio del array = " << N << endl;
	cout << "# Tiempo de CPU " << t.ms_elapsed << " mseconds" << endl;

#ifdef IMPRIMIR
	ofstream transformada_out("transformada_fftw.dat");
	for(int j = 0 ; j < Ncomp ; j++){
		COMPLEX Z = D_output[j];
		transformada_out << Z.real() << " " << Z.imag() << endl;
	}
#endif

	fftw_destroy_plan(plan_d2z);
#ifdef FFTWTHREADS
	fftw_cleanup_threads();
#endif
	return 0;

}

