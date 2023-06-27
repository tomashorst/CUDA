/*
=============================================================================================
A. B. Kolton para ICNPG2013, 30/10/2013
=============================================================================================

Este programita transforma Fourier una señal unidimensional real, almacenada en GPU, usando cufft.
Cronometra e imprime (opcionalmente) 
la transformada. 

Completando los TODO el programa deberia luego antitransformar la transformada recien obtenida, 
de modo de que Ud sea capaz de verificar la accion de las transformadas forward y backward, 
imprimiendo el array original y la antitransformada de la transformada...

Para compilar, simplemente:

$ nvcc simple_cufft_thrust_solucion.cu -lcufft -arch=sm_21 -o simple_cufft

o usando el Makefile provisto: 

$make
$make simple_fftw
$make simple_fftw_threads

La primera opcion compilara este codigo, y las dos siguientes
otros dos codigos, que son las versiones en CPU de este mismo (single and multiple thread respectivamente), 
pero usando FFTW en vez de CUFFT. Va a necesitar esta compilacion completa 
para hacer la comparacion GPU vs CPU.


OBJETIVOS:
- Practicar el manejo básico de la librería cuFFT (planes, ejecucion, input/output, layouts, complejos, etc).
- Practicar el manejo básico de la librería Thrust.
- Practicar la interoperabilidad cuFFT-Thrust.
- Evaluar performances de las distintas versiones del codigo en CPU y GPU.
- Repasar la matematica de la transformada de Fourier discreta.

EJERCICIOS:
- Utilizar simple y doble precisión, y diferentes tamaños. Comparar performances.
Para cambiar de precision hay que comentar/descomentar un #define DOUBLE_PRECISION mas abajo.

- Levantar los TODO y conteste las preguntas en el informe.

- Importante: Entender bien el output: ¿Cómo están ordenadas las frecuencias de la transformada? 
*/


/* algunos headers de la libreria thrust */
// https://github.com/thrust/thrust/wiki/Documentation
#include<thrust/transform.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <fstream>
#include "cutil.h"	// CUDA_SAFE_CALL, CUT_CHECK_ERROR
#include "gpu_timer.h"


// CUFFT include http://docs.nvidia.com/cuda/cufft/index.html
#include <cufft.h>

using namespace std;

/* + Array Size N: use only powers of 2 */
#ifndef TAMANIO
#define N 1048576
#else
#define N TAMANIO
#endif

#define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
typedef cufftDoubleReal REAL;
typedef cufftDoubleComplex COMPLEX;
#else
typedef cufftReal REAL;
typedef cufftComplex COMPLEX;
#endif

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
		// empiece con esta funcion, luego ponga cualquiera que quiera transformar...
		return A1*2.0*cosf(2*M_PI*tid*T1/(float)N) + A2*2.0*sinf(2*M_PI*tid*T2/(float)N);
    	}
};

#ifdef OMP
#include <omp.h>
#endif

///////////////////////////////////////////////////////////////////////////
int main(void) {

	// Un container de thrust para guardar el input real en GPU 
	thrust::device_vector<REAL> D_input(N);

	// toma el raw_pointer del array de input, para pasarselo a CUFFT luego
	REAL *d_input = thrust::raw_pointer_cast(&D_input[0]);

	// Un container de thrust para guardar el ouput complejo en GPU = transformada del input 
	int Ncomp=N/2+1;
	thrust::device_vector<COMPLEX> D_output(Ncomp);
	 

	// toma el raw_pointer del array de output, para pasarselo a CUFFT luego
	COMPLEX *d_output = thrust::raw_pointer_cast(&D_output[0]); 

	// crea el plan de transformada de cuFFT
	#ifdef DOUBLE_PRECISION
	cufftHandle plan_d2z;
	CUFFT_SAFE_CALL(cufftPlan1d(&plan_d2z,N,CUFFT_D2Z,1));
	#else
	cufftHandle plan_r2c;
	CUFFT_SAFE_CALL(cufftPlan1d(&plan_r2c,N,CUFFT_R2C,1));
	#endif

	// lleno array de tamanio N con la senial, a travez del functor "FillSignal"
	// thrust::make_counting_iterator() imita a un iterador sobre una secuencia 0,1,2,3,...
	// que no existe, para ahorrar memoria (implicit sequence). Ver documentacion en "fancy iterators"   	
	thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(N),D_input.begin(),FillSignal());

	/* ---- Start ---- */
	// un timer para GPU
	gpu_timer tfft;
	tfft.tic();
	//Transforma Fourier ejecutando el plan
	#ifdef DOUBLE_PRECISION
	CUFFT_SAFE_CALL(cufftExecD2Z(plan_d2z, d_input, d_output));
	#else
	CUFFT_SAFE_CALL(cufftExecR2C(plan_r2c, d_input, d_output));
	#endif
	tfft.tac();
	/* ---- Stop ---- */

	// declara un vector para copiar/guardar la transformada en el host:
	thrust::host_vector<COMPLEX> H_output=D_output;

	/* Imprime la transformada */
	cout << "# Tamanio del array = " << N << endl;
	cout << "# Tiempo de GPU " << tfft.ms_elapsed << " mseconds (fft)" << endl;

#ifdef IMPRIMIR
	ofstream transformada_out("transformada.dat");
	for(int j = 0 ; j < Ncomp ; j++){
		transformada_out << COMPLEX(H_output[j]).x << " " << COMPLEX(H_output[j]).y << endl;
	}
#endif


////////////////////////////////////////////////////////////////////
// TODO: Verifique matematicamente que el resultado sea correcto. Preste atencion al ordenamiento de las frecuencias...
// HINT: sin(x)=[e^{ix}-e^{-ix}]/2i , cos(x)=[e^{ix}+e^{-ix}]/2 y mire la formula de la antitransformada (clase)
// SOLUCION: 
// llamemos x=2*M_PI*tid/(float)N -> la senial es:
// A1*2.0*cosf(x*T1) + A2*2.0*sinf(x*T2) = A1*[e^{ixT1}+e^{-ixT1}] -i A2*[e^{ixT2}-e^{-ixT2}] = 
// A1*[e^{ixT1}+e^{-ixT1}] -i A2*[e^{ixT2}-e^{-ixT2}] 
// chequeo freqs: plot 'transformada.dat' u 2:0 w lp, 'transformada.dat' u 1:0 w lp, N/4, N/8
// chequeo amplitud: A1=4, A2=6 -> Re(T1)=A1*N, Im(T1)=0, Re(T2)=0, Im(T2)=A2*N
// en gnuplot:
// N=????; T1=N/4; T2=N/8; A1=N*4; A2=N*6; 
// plot 'transformada.dat' u 2:0 w lp, '' u 1:0 w lp, (0<x && x<A1)?(T1):(1./0) w p pt 8 lc 0, (0>x && x>-A2)?(T2):(1./0) w p pt 8 lc 0 

// TODO: 
// Agregue planes para realizar la antitransformada de la senial con CUFFT (contemple los casos double y float)
// SOLUCION:
	#ifdef DOUBLE_PRECISION
	// ....
	#else
	// ....
	#endif
	#ifdef DOUBLE_PRECISION
	cufftHandle plan_z2d;
	CUFFT_SAFE_CALL(cufftPlan1d(&plan_z2d,N,CUFFT_Z2D,1));
	#else
	cufftHandle plan_c2r;
	CUFFT_SAFE_CALL(cufftPlan1d(&plan_c2r,N,CUFFT_C2R,1));
	#endif

// TODO: 
// Declare/aloque un container de Thrust para guardar la antitransformada, y el raw_pointer para pasar a CUFFT
// SOLUCION:
	// Un container de thrust para guardar el input real en GPU 
	thrust::device_vector<REAL> D_anti(N);

	// toma el raw_pointer del array de input, para pasarselo a CUFFT luego
	REAL *d_anti = thrust::raw_pointer_cast(&D_anti[0]);

// TODO:
// Ejecute los planes cuFFT de la antitransformada (contemple los casos double y float)
	#ifdef DOUBLE_PRECISION
	//.....
	#else
	//.....
	#endif
	#ifdef DOUBLE_PRECISION
	CUFFT_SAFE_CALL(cufftExecZ2D(plan_z2d, d_output, d_anti));
	#else
	CUFFT_SAFE_CALL(cufftExecR2C(plan_c2r, d_output, d_anti));
	#endif

// TODO: 
// Declare/aloque dos containers de Thrust: uno para guardar la antitransformada, y otro para el input original, en el host
// y copie los respectivos contenidos del device al host
// SOLUCION:
//.....
//.....

// TODO:
// Imprima en un file el input original y la antitransformada de la transformada, para comparar
// SOLUCION:
#ifdef IMPRIMIR
	ofstream senial_out("senial_vs_antitransformada.dat");
	thrust::device_vector<REAL> H_anti(D_anti);
	thrust::device_vector<REAL> H_input(D_input);
	for(int j = 0 ; j < N ; j++){
		senial_out << H_input[j] << " " << H_anti[j]/N << endl;
	}
	//gnuplot > plot 'senial_vs_antitransformada.dat' u 0:1, '' u 0:2 w l
#endif
	return 0;
}

