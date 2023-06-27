from numba import cuda

@cuda.jit(device=True)
def compare_combinations(target, combination):
    # Convert the combination to a string
    combination_str = ""
    while combination > 0:
        remainder = combination % 62
        combination_str = chr(mapping_n2a(remainder)) + combination_str
        combination //= 62
    
    return combination_str == target

@cuda.jit
def find_combinations(target, target_length_max, attempts_max, result):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    stride = cuda.blockDim.x * cuda.gridDim.x

    for i in range(idx, attempts_max, stride):
        combination = i
        length = 0
        while combination > 0:
            combination //= 62
            length += 1
            
        if length > target_length_max:
            continue
        
        combination = i
        if compare_combinations(target, combination):
            result[0] = combination

def run_find_combinations(target, target_length_max, attempts_max):
    result = cuda.device_array(1, dtype=int)
    threads_per_block = 128
    blocks_per_grid = (attempts_max + (threads_per_block - 1)) // threads_per_block

    find_combinations[blocks_per_grid, threads_per_block](target, target_length_max, attempts_max, result)
    cuda.synchronize()

    if result[0] != 0:
        combination_str = ""
        combination = result[0]
        while combination > 0:
            remainder = combination % 62
            combination_str = chr(mapping_n2a(remainder)) + combination_str
            combination //= 62

        print("Match found in run_find_combinations:", combination_str)
    else:
        print("No match found in run_find_combinations")

def mapping_n2a(value): #Number to ASCII
    if value < 10:
        return value + 48            #numbers
    elif value < 36:
        return value + 65 - 10        #lowercase
    elif value < 62:
        return value + 97 - 10 - 26   #uppercase
    else:
        return -1

target = "2a4Be"  # Modify the target value to match
target_length_max = len(target)  # Length of combinations to compare
attempts_max = 1e9  # Maximum number of attempts

run_find_combinations(target, target_length_max, attempts_max)
