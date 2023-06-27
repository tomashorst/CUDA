using CUDA

function my_kernel(input_value::CuDeviceVector{Int})
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    num_elements = length(input_value)
    num_combinations = 2^num_elements

    if idx <= num_combinations
        combination = zeros(Int, num_elements)
        
        for i in 1:num_elements
            combination[i] = (idx >> (i - 1)) & 1
        end

        combination_value = sum(combination)
        
        if combination_value == input_value[1]
            return nothing
        end
    end
    return nothing
end

function compare_combinations(input_value::CuArray{Int})
    num_elements = length(input_value)
    num_combinations = 2^num_elements

    threads_per_block = 256
    blocks_per_grid = cld(num_combinations, threads_per_block)
    println("Corrio1")
    @cuda threads=threads_per_block blocks=blocks_per_grid my_kernel(input_value)
    synchronize()
    println("Corrio2")
    
    return output_value
end

input_value = CuArray{Int}([1, 2, 3])
result = compare_combinations(input_value)
println(result)
