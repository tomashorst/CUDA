import numpy as np
from numba import cuda, jit

# Define the CUDA kernel function to run in the other function
@cuda.jit
def kernel_mapping_n2a(value): #Number to ASCII
    if value < 10:
        return value + 48            #numbers
    elif value < 36:
        return value + 65 - 10        #lowercase
    elif value < 62:
        return value + 97 - 10 - 26   #uppercase
    else:
        return -1
    

@cuda.jit
def kernel_find_combinations(target_gpu, target_length_max, combination_gpu, result_gpu):
    # NroThreads_x_Blocks = 1024
    # NroBlocks_x_Grids = 233
    # NroGrids = 1
    # Lo haré sólo en x => NroThreadsTotales = 1024*233*1 = 238592 -> 2.4e5
    idx = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    #for i in range(idx, target_length_max):
    #    for j in range(stride, 62):
    # combination_gpu[x] = y
    # for k in range(x+1):
    #     combination_gpu[k] = kernel_mapping_n2a(combination_gpu[k])
    # if combination_gpu == target:
    #     result_gpu = combination_gpu
    #     return

def mapping_n2a(value):
    if value < 10:
        return chr(value + 48)  # numbers
    elif value < 36:
        return chr(value + 65 - 10)  # lowercase
    elif value < 62:
        return chr(value + 97 - 10 - 26)  # uppercase
    else:
        return ""

def run_find_combinations(target, target_length_max, result):
    block_size = np.int32(233)
    grid_size = np.int32(1)
    target_gpu = cuda.to_device(target)
    result_gpu = cuda.to_device(result)
    combination_gpu = cuda.to_device(np.zeros(len(target), dtype='int32'))
    kernel_find_combinations[grid_size, block_size](target_gpu, target_length_max, combination_gpu, result_gpu)
    cuda.synchronize()

    result = result_gpu.copy_to_host()
    if not np.array_equal(result, np.zeros(target_length_max, dtype='int32')):
        result = np.array([chr(char) for char in result], dtype=str)
        print("Match found in run_find_combinations:", result)
    else:
        print("No match found in run_find_combinations")

    

target = "2a4Be"  # Modify the target value to match
target_vector = np.array([ord(char) for char in target], dtype=np.int32)
target_length_max = np.int32(len(target))
#attempts_max = np.int32(1e9) 
result = np.zeros(len(target_vector), dtype='int32')
#result = np.array([ord(char) for char in target], dtype=np.int32)
run_find_combinations(target, target_length_max, result)
