// Compile with:
// nvcc --std=c++11 stream_parallel.cu -o stream_parallel -lcufft

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cufft.h>

// Print file name, line number, and error code when a CUDA error occurs.
#define check_cuda_errors(val)  __check_cuda_errors__ ( (val), #val, __FILE__, __LINE__ )

template <typename T>
inline void __check_cuda_errors__(T code, const char *func, const char *file, int line) {
    if (code) {
    std::cout << "CUDA error at "
          << file << ":" << line << std::endl
          << "error code: " << (unsigned int) code
          << " type: \""  << cudaGetErrorString(cudaGetLastError()) << "\"" << std::endl
          << "func: \"" << func << "\""
          << std::endl;
    cudaDeviceReset();
    exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {

    // Number of FFTs to compute.
    const int NUM_DATA = 64;

    // Length of each FFT.
    const int N = 1048576;

    // Number of GPU streams across which to distribute the FFTs.
    const int NUM_STREAMS = 4;

    // Allocate and initialize host input data.
    float2 **h_in = new float2 *[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        h_in[ii] = new float2[N];
        for (int jj = 0; jj < N; ++jj) {
            h_in[ii][jj].x = (float) 1.f;
            h_in[ii][jj].y = (float) 0.f;
        }
    }

    // Allocate and initialize host output data.
    float2 **h_out = new float2 *[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
    h_out[ii] = new float2[N];
    for (int jj = 0; jj < N; ++jj) {
            h_out[ii][jj].x = 0.f;
            h_out[ii][jj].y = 0.f;
        }
    }

    // Pin host input and output memory for cudaMemcpyAsync.
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        check_cuda_errors(cudaHostRegister(h_in[ii], N*sizeof(float2), cudaHostRegisterPortable));
        check_cuda_errors(cudaHostRegister(h_out[ii], N*sizeof(float2), cudaHostRegisterPortable));
    }

    // Allocate pointers to device input and output arrays.
    float2 **d_in = new float2 *[NUM_STREAMS];
    float2 **d_out = new float2 *[NUM_STREAMS];

    // Allocate intput and output arrays on device.
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        check_cuda_errors(cudaMalloc((void**)&d_in[ii], N*sizeof(float2)));
        check_cuda_errors(cudaMalloc((void**)&d_out[ii], N*sizeof(float2)));
    }

    // Create CUDA streams.
    cudaStream_t streams[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        check_cuda_errors(cudaStreamCreate(&streams[ii]));
    }

    // Creates cuFFT plans and sets them in streams
    cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*NUM_STREAMS);
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        cufftPlan1d(&plans[ii], N, CUFFT_C2C, 1);
        cufftSetStream(plans[ii], streams[ii]);
    }

    // Fill streams with async memcopies and FFTs.
    for (int ii = 0; ii < NUM_DATA; ii++) {
        int jj = ii % NUM_STREAMS;
        check_cuda_errors(cudaMemcpyAsync(d_in[jj], h_in[jj], N*sizeof(float2), cudaMemcpyHostToDevice, streams[jj]));
        cufftExecC2C(plans[jj], (cufftComplex*)d_in[jj], (cufftComplex*)d_out[jj], CUFFT_FORWARD);
        check_cuda_errors(cudaMemcpyAsync(h_out[jj], d_out[jj], N*sizeof(float2), cudaMemcpyDeviceToHost, streams[jj]));
    }

    // Wait for calculations to complete.
    for(int ii = 0; ii < NUM_STREAMS; ii++) {
        check_cuda_errors(cudaStreamSynchronize(streams[ii]));
    }

    // Free memory and streams.
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        check_cuda_errors(cudaHostUnregister(h_in[ii]));
        check_cuda_errors(cudaHostUnregister(h_out[ii]));
        check_cuda_errors(cudaFree(d_in[ii]));
        check_cuda_errors(cudaFree(d_out[ii]));
        delete[] h_in[ii];
        delete[] h_out[ii];
        check_cuda_errors(cudaStreamDestroy(streams[ii]));
    }

    delete plans;

    cudaDeviceReset();  

    return 0;
}
