#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q cpu.q

## Pido N slots en una misma maquina, pruebe cambiar el numero...
#$ -pe neworte 14

## Limito los threads a cantidad de slots pedidos
export OMP_NUM_THREADS=$NSLOTS

#ejecuto los binarios

echo "=================="
echo "numero de threads omp = " $OMP_NUM_THREADS
hostname
time ./omp.out
echo "=================="
