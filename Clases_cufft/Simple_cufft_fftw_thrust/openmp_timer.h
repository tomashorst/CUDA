///////////////////////////////// OPENMP TIMER ////////////////////////////////
// cronometro para procesos en CPU multithread con openMP
// USO: omp_timer T; T.tic();...calculo...;T.tac(); cout << T.ms_elapsed << "ms\n";

#pragma once

#include <omp.h>
#include <time.h>
#include <math.h>
#include <stdio.h>


struct omp_timer
{
//  clock_t start;
//  clock_t end;
  double wall_timer_start;
  double wall_timer_end;
  float ms_elapsed;

  omp_timer(void)
  {
    tic();
  }

  ~omp_timer(void)
  {
  }

  void tic(void)
  {    	
//    start = clock();
    wall_timer_start = omp_get_wtime(); 
  }

  double tac(void)
  {

//    end = clock();
    wall_timer_end = omp_get_wtime(); 
//    return static_cast<double>(end - start) / static_cast<double>(CLOCKS_PER_SEC);
    return (ms_elapsed=1e3*static_cast<double>(wall_timer_end - wall_timer_start));
  }

  double epsilon(void)
  {
    return 1.0 / static_cast<double>(CLOCKS_PER_SEC);
  }
};

#define CRONOMETRAR_OMP( X,VECES ) {  { \
                            omp_timer t; \
			    float msacum=0.0;\
			    float msacum2=0.0;\
			    for(int n=0;n<VECES;n++){\
			    	t.tic();\
                            	X; t.tac();\
				msacum+=t.ms_elapsed;\
				msacum2+=(t.ms_elapsed*t.ms_elapsed);\
			    }\
			    std::cout << "OMP: " << (msacum/VECES) << " +- " << \
			    (sqrt(msacum2/VECES - msacum*msacum/VECES/VECES)) \
			    << " ms (" << VECES << " veces, " << omp_get_max_threads() << " threads)\n" ; \
                            }}




