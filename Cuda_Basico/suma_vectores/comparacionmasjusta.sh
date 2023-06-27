#compilamos NVECES=1
nvcc 6_suma_vectores_gpu_masgeneral.cu -o gpu -DNVECES=1; 
g++ 0_suma_vectores_cpu.cpp -DNVECES=1 -o cpu

time ./gpu 10000000; 

time ./cpu 10000000

nvcc 6_suma_vectores_gpu_masgeneral.cu -o gpu -DNVECES=1000; 
g++ 0_suma_vectores_cpu.cpp -DNVECES=1000 -o cpu

time ./gpu 10000000; 

time ./cpu 10000000
