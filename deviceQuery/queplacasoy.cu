#include <stdio.h>

int main(int argc, char **argv)
{
	cudaDeviceProp deviceProp;

	int deviceCount = 0;
    	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);


	printf("En este nodo hay %d placas\n\n",deviceCount);
	for(int dev=0;dev<deviceCount;dev++){
	    	cudaSetDevice(dev);
    		cudaGetDeviceProperties(&deviceProp, dev);
    		printf("Hola!, yo soy [Device %d: \"%s\"], tu acelerador grafico personal\n", dev, deviceProp.name);
	}

	int dev; cudaGetDevice(&dev);
	printf("\nle asigno la device %d, que esta desocupada\n", dev);

	return 0;
}
