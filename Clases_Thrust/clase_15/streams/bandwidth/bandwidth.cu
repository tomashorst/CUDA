#include <cstdio>
#include <sys/time.h>

#include "helpers_cuda.h"

static const size_t MAX_MEM = 1UL << 28;    // 4 x 256MB

char * device_source;
char * device_dest;
char * host_source;
char * host_dest;


static void alloc_buffers(void)
{
    CHECK_CUDA_CALL(cudaMalloc(&device_source, MAX_MEM));
    CHECK_CUDA_CALL(cudaMalloc(&device_dest, MAX_MEM));

    host_source = new char[MAX_MEM];
    host_dest = new char[MAX_MEM];
}


static void free_buffers(void)
{
    CHECK_CUDA_CALL(cudaFree(device_source));
    CHECK_CUDA_CALL(cudaFree(device_dest));

    delete[] host_source;
    delete[] host_dest;
}


int main(int argc, char ** argv)
{
    struct timeval start, finish, elapsed;

    alloc_buffers();

    for (size_t bytes = 1; bytes <= MAX_MEM; bytes *= 2) {
        gettimeofday(&start, NULL);
        CHECK_CUDA_CALL(cudaMemcpy(device_dest, host_source, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_CALL(cudaMemcpy(host_dest, device_source, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA_CALL(cudaDeviceSynchronize());
        gettimeofday(&finish, NULL);

        timersub(&finish, &start, &elapsed);
        double ms = elapsed.tv_usec / 1000.0 + elapsed.tv_sec * 1000.0;
        double GBps = (double) bytes * 2.0 * 1000.0 / (ms * (double) (1UL << 30));
        printf("%lu bytes in %.3lf ms (%.3lf GBps)\n", bytes, ms, GBps);
    }

    free_buffers();

    return 0;
}
