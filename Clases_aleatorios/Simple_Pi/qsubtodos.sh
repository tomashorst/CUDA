for i in *_Pi; do echo $i; qsub -N $i jobGPU "./"$i;done
