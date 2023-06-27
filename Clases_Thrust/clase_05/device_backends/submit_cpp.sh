#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q cpu.q

#ejecuto los binarios

echo "=================="
echo "numero de threads = 1"
hostname
time ./cpp.out
echo "=================="
