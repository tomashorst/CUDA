/*
 * heat.cuh
 *
 *  Created on: May 16, 2012
 *      Author: Eze Ferrero, Ale Kolton
 */

/*
Este programa resuelve la ecuacion de diffusion en 2D
con condiciones de contorno periodicas de contorno
en funcion del tiempo, con una fuente definida por el usuario.

\partial_t phi(x,y,t) = (\partial^2_x + \partial^2_y) phi(x,y,t) + \rho(x,y,t)

- Usa cufft para desacoplar (diagonalizar) las ecuaciones
- Pasos de Euler en espacio de Fourier para evolucionar un Dt en el tiempo


==============
Objetivo:
Usar la CUFFT para una aplicacion concreta: resolver la ecuacion de arriba en diferencias finitas, 
aprovechando que queda desacoplada en espacio de Fourier (i.e. los modos de Fourier evolucionan independientemente). 

==============
Que hacer:

* Compilar y hacer que el codigo corra correctamente completando heat_template.cuh en los 10 "FIXME" con instrucciones que contiene, y renombrandolo como heat.cuh. 

* Mas instrucciones en el enunciado de la Guia.
==============

FILES:

main.cu: contiene la resolucion general iterativa para la evolucion en el tiempo del campo de temperaturas y se encarga de imprimir imagenes ppm del campo de temperaturas, y permite al usuario modificar en funcion del tiempo las fuentes externas en el espacio real.

ppmimages.cuh: contiene funciones para produccion de imagenes ppm a partir del campo de temperaturas.

heat_template.cuh: contiene todas las declaraciones y definiciones de las variables y funciones necesarias para iterar un paso de tiempo el campo de temperaturas. 

Makefile: ademas de sistematizar el proceso de compilacion, contiene los parametros constantes de la simulacion.

==============

*/

#include <cmath>
#include <assert.h>
#include <sys/time.h>   /* gettimeofday */

#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_timer.h"
//#include "curso.h"
#include <cufft.h>

#include <stdio.h>      /* print */
#include <stdlib.h>
#include <iostream>     /* print */
#include <iomanip>      /* print */
#include <fstream>      /* print */


// PARAMETERS

#ifndef LX
#define LX 512                  /* System's X size*/
#endif

#ifndef LY
#define LY 512                  /* System's Y size*/
#endif

#ifndef T_RUN
#define T_RUN 1000              /* Running time */
#endif

#ifndef T_DATA
#define T_DATA 10              /* Data acquisition time */
#endif

#ifndef DT
#define DT 0.05f                 /* Integration step time */
#endif

#ifndef C_BETA
#define C_BETA 1.0f             /* Laplacian prefactor */
#endif

// Hardware parameters for GTX 470/480 (GF100)
#define SHARED_PER_BLOCK 49152
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 65535

#define TILE_X 8        // each block of threads is a tile
#define TILE_Y 32       // each block of threads is a tile

// Including our declarations
#include "heat.cuh"

/////////////////////////////////////////////////////////
int main(){
        assert(TILE_X%2==0);
        assert(TILE_Y%2==0);
        assert(LX%TILE_X==0);
        assert(LY%TILE_Y==0);
        assert(T_RUN%T_DATA==0);
        assert(DT<=0.5);        //We've assumed a small temporal step in the integration scheme

        // Choosing less "shared memory", more cache.
        //CUDA_SAFE_CALL(cudaThreadSetCacheConfig(cudaFuncCachePreferL1)); 


        // Print header
        printf("# Lx: %i\n", LX);
        printf("# Ly: %i\n", LY);
        printf("# Running time (in Euler steps): %i\n", T_RUN);
        printf("# Data Acquiring Step: %i\n", T_DATA);
        printf("# beta: %f\n", C_BETA);
        printf("# DT for Euler step: %f\n", DT);

        // main{} variables
        char filename [200];

        // Our class
        heat_model T;

        /* Calculate discretized Fourier modes */
        T.SetQxQy();

        /* Choose a particular initial condition for the fields: phi (quien=0) or rho(x,y) (quien=1) */
        // USE: InitParticular(quien, Lx, Ly, shape, x_offset in (-.5,.5), y_offset in (-.5,.5), width in (0,1) prop to LY, value) 
        // shape: 0=uniforme, 1=band, 2=circle, 3=square_with_hole, ..... more to come	
	T.InitParticular(0, LX,LY,0,0.1,0.1,0.,0.);   // perfil de tempeatura inicial
	T.InitParticular(1, LX,LY,0,0.2,0.2,0.1,1.0); // perfil de fuentes inicial 

        /* Take the scalar field and the force to Fourier Space */
        T.TransformToFourierSpace(); 
        T.TransformForceToFourierSpace(); 

	gpu_timer cronometro;
	cronometro.tic();
        for(int i=0;i<T_RUN+1; i++){

                /* Get data each T_DATA steps*/
                if(i%T_DATA==0) {
                        /* Take the scalar field back from Fourier space and normalize */
                        T.AntitransformFromFourierSpace();
                        T.Normalize();

                        T.CpyDeviceToHost(); // To be used in a moment for printing.
 
                        /* Print frame for visualization */
                        sprintf(filename, "frame%d.ppm", 100000000+i);
                        ifstream file(filename);
                        T.PrintPicture(filename,i);
                }
		{
			float phase=2.f*M_PI*i*4.f/T_RUN;
			float x0=0.2*sin(phase);float y0=0.2*cos(phase); float intensidad=0.1;
			T.InitParticular(1, LX,LY,4,x0,y0,0.1,intensidad); // perfil de fuentes instantaneo			
		        T.TransformForceToFourierSpace(); // transforma perfil de fuentes para Euler Step
		}	

                /* Integration step in Fourier Space*/
                T.EulerStep();               
        }
	cronometro.tac();

        printf("# System size: LX=%i LY=%i . heatFFT execution time (in msecs): %lf \n", LX, LY, cronometro.ms_elapsed);

        return 0;
}

