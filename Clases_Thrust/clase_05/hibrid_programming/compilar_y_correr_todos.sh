for i in farber1-11_async  farber1-11_sync
do 
	nvcc -Xcompiler -fopenmp $i.cu -o $i.out 	
	#qsub -v prog="$i.out" submit.sh
	echo $i
	./$i.out 5000000
done


