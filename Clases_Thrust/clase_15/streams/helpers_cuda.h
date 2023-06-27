#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


// Helper for CHECK_CUDA_CALL
static inline void _checkCudaReturnValue(cudaError_t result,
    char const * const func,
    const char * const file,
    int const line) 
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, (unsigned int) result, cudaGetErrorString(result), func);
        exit((unsigned int) result);
    }
}


// Check the result of a CUDA API call and abort with information on error
#define CHECK_CUDA_CALL(f) _checkCudaReturnValue( (f), #f, __FILE__, __LINE__ )
