# compilar codigo thrust para omp device backend
# uso: openmpbackend.sh nombredelcusinextension

comando="cp $1.cu $1.cpp"
echo $comando
$comando


# Asi compilamos thrust sin usar nvcc, para device backend = openmp (paralelo en CPU multicore)
comando="g++ -O2 $1.cpp -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -I/usr/local/cuda/include/ -o ejecutable_omp"
echo $comando
$comando


# Asi compilamos thrust sin usar nvcc, para device backend = CPU (secuencial)
comando="g++ -O2 $1.cpp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -I/usr/local/cuda/include/ -o ejecutable_cpp"
echo $comando
$comando


# Asi compilamos thrust con nvcc, para device backend = CUDA (paralelo en GPU)
comando="nvcc -O2 $1.cu -o ejecutable_cuda -Wno-deprecated-gpu-targets"
echo $comando
$comando


