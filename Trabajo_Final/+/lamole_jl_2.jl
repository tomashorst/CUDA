
using CUDA

function compare_combinations_kernel(input_value, result)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # Generar combinaciones posibles
    num_elements = length(input_value)
    num_combinations = 2^num_elements

    if idx <= num_combinations
        combination = zeros(Int, num_elements)
        
        # Generar combinación actual
        for i in 1:num_elements
            combination[i] = (idx >> (i - 1)) & 1
        end

        # Calcular el valor resultante de la combinación
        combination_value = sum(combination)
        
        # Comparar el valor resultante con el valor de entrada
        result[idx] = combination_value == input_value[1]
    end
    return nothing
end

function compare_combinations(input_value::Vector{Int})
    println("Corrio2...")
    num_elements = length(input_value)
    num_combinations = 2^num_elements

    input_value_d = CUDA.CuArray(input_value)
    result_h = similar(input_value, Bool)
    
    threads_per_block = 256
    blocks_per_grid = cld(num_combinations, threads_per_block)
    
    result_d = similar(result_h, Bool)
    
    @cuda threads=threads_per_block blocks=blocks_per_grid compare_combinations_kernel(input_value_d, result_d)
    println("Corrio3...")
    CUDA.synchronize()
    result_h .= result_d
    println("Corrio4...")
    return result_h
end

println("Corrio1...")

result = compare_combinations(input_value)

println("Corrio5...")
println(result)
