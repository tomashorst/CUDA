using CUDA

function my_kernel(x::CuDeviceVector{Float32}, r::CuDeviceVector{Float32},l::CuDeviceVector{Float32},n_iterations)
	i = (blockIdx().x-1) * blockDim().x + threadIdx().x

	acum=Float32(0.0);
	for p in 1:n_iterations
		@inbounds x[i] = r[i] * x[i] * (1 - x[i])
		@inbounds acum = acum + log(abs(r[i]-2.0*r[i]*x[i]) )
	end

	l[i]=acum/n_iterations;

	return nothing
end


# Define the parameters
N = 10000 #number of variables

r_min = Float32(3.5)
r_max = Float32(4.0)
r_range = range(r_min, r_max, length=N)  # range of logistic parameter

n_iterations = 10000 # number of map iterations


# Allocate memory on the GPU
x_d = CUDA.rand(Float32, N) # variables
r_d = CuArray{Float32}(r_range) # parameters
l_d = CUDA.zeros(Float32, N)  # Lyapunov exponent

# Alternative1: Iterate the logistic equation on the GPU using one raw kernel
#@cuda threads=100 blocks=100 my_kernel(x_d,r_d,l_d,n_iterations)

# Alternative2: Iterate the logistic equation on the GPU using many broadcast kernels
for p in 1:n_iterations
	global x_d, r_d, l_d
	CUDA.@. x_d  = (r_d .* x_d) .* (1.0 .- x_d)
        CUDA.@. l_d = l_d .+ log( abs(r_d.-2*r_d.*x_d) ) 
end



# Copy the results back to the CPU	
x_range = Array{Float32}(x_d)
l_range = Array{Float32}(l_d)


using DelimitedFiles
data = hcat(r_range, x_range, l_range)
# Save the matrix as a two-column file
writedlm("logis.txt", data, '\t')

# Plot the results
#using Plots
#col = [xi < 0 ? "blue" : "red" for xi in l_range]
#scatter(r_range, x_range, color=col, markersize=1, legend=false, markerstrokecolor=col);
#xlabel!("r")
#ylabel!("x")

#savefig("myplot.png")













