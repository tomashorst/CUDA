#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
##  pido la cola gpu.q
##$ -q gpu.q@compute-0-2
#$ -q gpu.q
## pido una placa
#$ -l gpu=1
#
#ejecuto el binario

#/usr/local/cuda-5.5/bin/nvprof ./simple_cufft
./simple_cufft
