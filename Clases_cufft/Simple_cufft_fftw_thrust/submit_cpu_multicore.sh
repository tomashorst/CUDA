#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
##  pido la cola cpu.q
#$ -q cpu.q
#ejecuto el binario

./simple_fftw_threads
