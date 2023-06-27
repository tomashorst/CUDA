#compilamos para gpu
nvcc 6_suma_vectores_gpu_masgeneral.cu -o gpu -DNVECES=1; 

#conpilamos para cpu
g++ 0_suma_vectores_cpu.cpp -DNVECES=1 -o cpu

# largamos distintos tamanos
for l in 10 100 1000 10000 100000 1000000 10000000 100000000
do 
	echo $(./gpu $l | awk '{print $6,$8}') $(./cpu $l | awk '{print $5}'); 
done > zzz

#mostramos los numeros
cat zzz

#graficamos
#gnuplot -e "set size ratio 2; set multi lay 1,2; set logs; set xla 'N'; set yla 'tiempo [ms]';  plot 'zzz' u 1:2 w lp t 'gpu', '' u 1:3 w lp t 'cpu'; set yla 'speedup'; p 'zzz' u 1:(\$3/\$2) w lp t '';" --persist


