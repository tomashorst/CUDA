/*
 * Imprime "hola mundo" desde 10 hilos corriendo en GPU
 */


#include "common.h"
#include <stdio.h>

__global__ void helloFromGPU()
{
    printf("hola mundo desde la GPU!\n");
}


int main(int argc, char **argv)
{
    printf("Hola mundo desde la CPU!\n\n");

    helloFromGPU<<<1, 10>>>();

    CHECK(cudaDeviceSynchronize());
    return 0;
}


