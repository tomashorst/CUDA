# compilar codigo thrust para omp device backend
# uso: openmpbackend.sh nombredelcusinextension

comando="cp $1.cu $1.cpp"
echo $comando
$comando

comando="g++ -O2 $1.cpp -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -I/usr/local/cuda-6.5/include/"
echo $comando
$comando
