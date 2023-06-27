for i in *_miniising; do echo $i; qsub -N $i jobGPU "./"$i;done
