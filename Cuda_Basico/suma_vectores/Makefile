APPS=a.out
#0_suma_vectores_cpu \
#1_suma_vectores_gpu_1bl1024th_naive \
#2_suma_vectores_gpu_1bl1024th_serializado \
#3_suma_vectores_gpu_muchos_bloques_un_thread_por_bloque \
#4_suma_vectores_gpu_general \
#5_suma_vectores_gpu_thrust

N ?= 1024 

#all: ${APPS}

0:	0_suma_vectores_cpu.cpp
	g++ -O4 0_suma_vectores_cpu.cpp		

1:	1_suma_vectores_gpu_1bl1024th_naive.cu
	nvcc -Wno-deprecated-gpu-targets 1_suma_vectores_gpu_1bl1024th_naive.cu 		

2:	2_suma_vectores_gpu_1bl1024th_serializado.cu
	nvcc -Wno-deprecated-gpu-targets 2_suma_vectores_gpu_1bl1024th_serializado.cu 		

3: 	3_suma_vectores_gpu_muchos_bloques_un_thread_por_bloque.cu
	nvcc -Wno-deprecated-gpu-targets 3_suma_vectores_gpu_muchos_bloques_un_thread_por_bloque.cu 

4:	4_suma_vectores_gpu_general.cu
	nvcc -Wno-deprecated-gpu-targets -O4 4_suma_vectores_gpu_general.cu		

5: 	5_suma_vectores_gpu_thrust.cu
	nvcc -Wno-deprecated-gpu-targets 5_suma_vectores_gpu_thrust.cu

6:	6_suma_vectores_gpu_masgeneral.cu
	nvcc -Wno-deprecated-gpu-targets -O4 6_suma_vectores_gpu_masgeneral.cu		


clean:
	rm -f ${APPS}

run:	
	nvprof ./a.out $N
	#./a.out $N

submit:	
	qsub jobGPU $N; watch qstat


run0:	clean 0 run		

run1:	clean 1 run		

run2:	clean 2 run		

run3: 	clean 3 run		

run4:	clean 4 run		

run5:	clean 5 run		

run6:	clean 6 run		


qsub0:	clean 0 submit		

qsub1:	clean 1 submit		

qsub2:	clean 2 submit		

qsub3: 	clean 3 submit		

qsub4:	clean 4 submit		

qsub5:	clean 5 submit		

qsub6:	clean 6 submit		


