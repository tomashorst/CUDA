import numpy as np
from numba import cuda, jit
#from numba.cuda.random import create_xoroshiro128p_states

# Define the CUDA kernel function to run in the other function
@cuda.jit
def mapping_n2a(value): #Number to ASCII
    if value < 10:
        return value + 48            #numbers
    elif value < 36:
        return value + 65 - 10        #lowercase
    elif value < 62:
        return value + 97 - 10 - 26   #uppercase
    else:
        return -1

# Define the CUDA kernel function
@cuda.jit
def compare_random_kernel(target, target_length_max, attempts_max, result):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x

    for index in range(idx, attempts_max, stride):
        attempts = 0
        while attempts < attempts_max:
            attempts += 1
            #combination = ""
            seed = mapping_n2a(idx)
            #combination = create_xoroshiro128p_states(seed=1)
            #combination = numba.cuda.random.create_xoroshiro128p_states(target_length_max, seed, subsequence_start=0, stream=0)
            #No se si está el % 
            #for i in range(target_length_max):
            #    combination[i] = combination[i] % 61
            #    combination[i] = int(combination[i])
            #    combination[i] += mapping_n2a(combination[i])
# it is important to make sure that each thread has its own RNG state, and they have been initialized to produce non-overlapping sequences
#Numba (like cuRAND) uses the Box-Muller transform to generate normally distributed random numbers.
    #       if combination == target:
    #            result = combination
    return



    #characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def compare_random(target, target_length_max, attempts_max, result):
    block_size = 128
    grid_size = int((attempts_max + block_size - 1) // block_size)

    # Transfer variables to GPU
    target_gpu = cuda.to_device(target)
    result_gpu = cuda.to_device(result)

    # Launch the CUDA kernel
    compare_random_kernel[grid_size, block_size](target_gpu, target_length_max, attempts_max, result_gpu)

    # Transfer the results back to the CPU
    result = result_gpu.copy_to_host()
    if(result != ""):
        print("Match found in compare_random:", result)
    else:
        print("No match found in compare_random")

    
target = "2a4Be"  # Modify the target value to match
target_length_max = len(target)  # Length of combinations to compare
attempts_max = 1e9  # Maximum number of attempts    
result = ""
compare_random(target, target_length_max, attempts_max, result)
