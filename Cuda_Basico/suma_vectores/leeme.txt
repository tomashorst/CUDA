##Instrucciones para hacer un profiling muy simple:

=================================================
##CPU (C/C++)
##gprof

g++ -pg suma_vectores_cpu.cpp -o a.out; 
./a.out; gprof ./a.out

=================================================
##GPU (CUDA)
##nvprof & nvvp
##https://devblogs.nvidia.com/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/

nvcc  suma_vectores_gpu.cu -o a.out; 

##Modo Texto: 
nvprof ./a.out (texto)

##Modo Visual (solo si tienen el cuda-toolkit instalado en sus maquinas): 
nvprof  -o out.prof ./a.out; nvvp out.prof

=================================================
##CPU y GPU (runtime)
##Incluir cpu_timer.h y gpu_timer.h y usar como esta en los ejemplos.

....
gpu_timer Reloj;
Reloj.tic();
.... (hacer algo en gpu)....
printf("N= %d t= %lf ms\n", N, Reloj.tac());

##cambiar gpu_timer por cpu_timer para CPU.
=================================================
