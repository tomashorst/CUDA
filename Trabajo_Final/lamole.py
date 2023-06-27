import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
    
@cuda.jit
def kernel_generate_combinations(match, rng_states, target_vector, result, characters, n):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Calculate the overall index of the thread
    #cuda.local.array: no deja pasarle n como parámetro, hacer ifs...
    if n == 1:
        candidate = cuda.local.array(1, 'int32')
    elif n == 2:
        candidate = cuda.local.array(2, 'int32')
    elif n == 3:
        candidate = cuda.local.array(3, 'int32')
    elif n == 4:
        candidate = cuda.local.array(4, 'int32')
    elif n == 5:
        candidate = cuda.local.array(5, 'int32')
    elif n == 6:
        candidate = cuda.local.array(6, 'int32')
    elif n == 7:
        candidate = cuda.local.array(7, 'int32')
    elif n == 8:
        candidate = cuda.local.array(8, 'int32')
    elif n == 9:
        candidate = cuda.local.array(9, 'int32')
    elif n == 10:
        candidate = cuda.local.array(10, 'int32')
    elif n == 11:
        candidate = cuda.local.array(11, 'int32')
    elif n == 12:
        candidate = cuda.local.array(12, 'int32')
    for i in range(0, n, 1):
        random_index = cuda.random.xoroshiro128p_uniform_float32(rng_states, idx)  # Generate a random index in the range [0, 1)
        character_index = int(random_index * len(characters))  # Map the random index to the valid range
        candidate[i] = characters[character_index]
        if not (candidate[i] == target_vector[i]):
            match = 5
    if match == 1:    
        for k in range(0, n, 1):
            result[k] = candidate[k]
        return

def generate_combinations(target_vector, n, characters):

    num_threads_per_block = 1024
    print("num_threads_per_block: ", num_threads_per_block)
    #num_blocks = int(2147483647)
    num_blocks = 100000
    print("num_blocks: ", num_blocks)
    rng_states = create_xoroshiro128p_states( num_threads_per_block * num_blocks, seed=1)
    characters_gpu = cuda.to_device(characters)
    result_gpu = cuda.to_device(np.zeros(n, dtype='int32'))
    target_vector_gpu = cuda.to_device(target_vector)
    match = np.int32(1)
    kernel_generate_combinations[num_blocks, num_threads_per_block](match, rng_states, target_vector_gpu, result_gpu, characters_gpu, n)
    cuda.synchronize()
    result = result_gpu.copy_to_host()
    if np.array_equal(result, target_vector):
        result = np.array([chr(char) for char in result], dtype=str)
        print("Match found in run_find_combinations:", result)
    else:
        print("No match found in run_find_combinations")
    return 


target = "2a4tX"  
n = np.int32(5) # len(target)
target_vector = np.array([ord(char) for char in target], dtype='int32')
characters = np.array([ord(char) for char in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"], dtype='int32')
#num_corridas = int(62**n)
generate_combinations(target_vector, n, characters)

