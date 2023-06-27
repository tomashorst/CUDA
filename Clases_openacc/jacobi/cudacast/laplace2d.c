/*
 *  Copyright 2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include<omp.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <openacc.h>
#include "timer.h"


#define NN 4096 //256
#define NM 4096 //256

double A[NN][NM];
double Anew[NN][NM];

void imprimir();

int main(int argc, char** argv)
{
    const int n = NN;
    const int m = NM;
    const int iter_max = 1000;
    
    const double tol = 1.0e-6;
    double error     = 1.0;
    
    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));
        
    // condiciones de borde 
    for (int j = 0; j < n; j++)
    {
        A[j][0]    = 1.0+sin(5.0*j*M_PI/NN);
        Anew[j][0] = 1.0+sin(5.0*j*M_PI/NN);
        A[0][j]    = 1.0;
        Anew[0][j] = 1.0;
    }

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    #if _OPENACC
    acc_init(acc_device_nvidia);
    #endif

    #if _OPENMP // solo para chequar cuandos openmp threads tenemos
    #pragma omp parallel
    {int tid = omp_get_thread_num();printf("Hello World from thread = %d\n", tid);}
    #endif

    StartTimer();
    int iter = 0;
    
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

#pragma omp parallel for shared(m, n, Anew, A)
#pragma acc kernels
        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                Anew[j][i] = 0.25 * ( A[j][i+1] + A[j][i-1]
                                    + A[j-1][i] + A[j+1][i]);
                error = fmax( error, fabs(Anew[j][i] - A[j][i]));
            }		
        }
        
#pragma omp parallel for shared(m, n, Anew, A)
#pragma acc kernels
        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                A[j][i] = Anew[j][i];    
            }
        }

        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

    double runtime = GetTimer();
 
    printf(" total: %f s\n", runtime / 1000);
	
    //imprimir();	
}

void imprimir()
{
    FILE *fout=fopen("matriz.dat","w");
    const int n = NN;
    const int m = NM;

    for( int j = 1; j < n-1; j++)
    {
    	for( int i = 1; i < m-1; i++ )
        {
        	fprintf(fout,"%f ",A[j][i]);    
        }
       	fprintf(fout,"\n");    
    }
}

