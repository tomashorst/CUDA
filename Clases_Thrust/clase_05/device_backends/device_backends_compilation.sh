nvcc -Xcompiler -fopenmp device_backends.cu -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -o cpp.out
nvcc -Xcompiler -fopenmp device_backends.cu -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -o cuda.out
nvcc -Xcompiler -fopenmp device_backends.cu -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -o omp.out

qsub submit_cuda.sh
qsub submit_omp.sh
qsub submit_cpp.sh

