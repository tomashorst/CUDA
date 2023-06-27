# Paths & Includes
INCLUDES 		+= -I/state/partition1/apps/spack/opt/spack/linux-centos7-x86_64/gcc-8.2.0/cuda-8.0.61-775ideynz7rzyz5mhv4rpdptuu4px4f5/samples/common/inc/

# Flags
NVCC_FLAGS              = -Wno-deprecated-gpu-targets

NVCC                    = nvcc
GCC			= g++
LINKER			= $(NVCC)
LDFLAGS			= -lcufft 


# Rules
all: clean async

async: async.cu
	$(LINKER) $(NVCC_FLAGS) -o async async.cu 

asyncthrust: async_thrust.cu
	$(LINKER) $(NVCC_FLAGS) -o asyncthrust async_thrust.cu -DTHRUST18

cufftstreams: cufft_streams.cu
	$(LINKER) $(NVCC_FLAGS) -o cufftstreams cufft_streams.cu -lcufft

nvtxstreams: nvtxstreams.cu
	$(LINKER) $(NVCC_FLAGS) -o nvtxstreams nvtxstreams.cu -DUSE_NVTX -lnvToolsExt 

pingpong: ping_pong.cu
	$(LINKER) $(NVCC_FLAGS) -o pingpong ping_pong.cu

muysimple: thrust_streams_muy_simple.cu
	$(LINKER) $(NVCC_FLAGS) -o muysimple thrust_streams_muy_simple.cu
	  
clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log ejecutable async asyncthrust cufftstreams pingpong nvtxstreams muysimple *.prof

cleanoutputs:
	rm -f async.o* cufftstreams.o* pingpong.o* nvtxstreams.o* asyncthrust.o* muysimple.o* *.prof
                                                     
run_async: async
	mv async ejecutable; qsub -N "async" jobGPU

run_cufftstreams: cufftstreams
	mv cufftstreams ejecutable; qsub -N "cufftstreams" jobGPU

run_asyncthrust: asyncthrust
	mv asyncthrust ejecutable; qsub -N "asyncthrust" jobGPU

run_nvtxstreams: nvtxstreams
	mv nvtxstreams ejecutable; qsub -N "nvtxstreams" jobGPU

run_pingpong: pingpong
	mv pingpong ejecutable; qsub -N "pingpong" jobGPU

run_muysimple: muysimple
	mv muysimple ejecutable; qsub -N "muysimple" jobGPU


