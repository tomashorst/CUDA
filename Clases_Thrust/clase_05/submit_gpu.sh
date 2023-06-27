#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
##  pido la cola gpu.q
#$ -q gpu.q
## pido una placa
#$ -l gpu=1
#
#ejecuto el binario

echo "en alguna GPU..."
hostname
time ./a.out
