#corre todos lo  backends de reduce

make cuda_submit APP=reduce 
make omp_submit APP=reduce 
make cpp_submit APP=reduce

watch qstat
