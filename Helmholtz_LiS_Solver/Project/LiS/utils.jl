using FFTW
using SparseArrays
using LinearAlgebra


function getKernelOperator(kernel::Array{ComplexF64}, n::Vector{Int64})
    k_n = size(kernel, 1) # the kernel is of size (kernel_n,kernel_n)
    kernel_op = zeros(ComplexF64,n...)
    kernel_op[1:k_n-1,1:k_n-1] = kernel[2:end,2:end]
    kernel_op[end,1:k_n-1] = kernel[1,2:end]
    kernel_op[1:k_n-1,end] = kernel[2:end,1]
    kernel_op[end,end] = kernel[1,1]
    return kernel_op
end

# function getLaplace1D(n::Int64, h::Float64, matrix_entry)
#     Lap1D = spdiagm(0=>(2/h^2)*ones(ComplexF64, n),1=>(-1/h^2)*ones(ComplexF64, n-1),-1=>(-1/h^2)*ones(ComplexF64, n-1))
#     Lap1D[1,1] = Lap1D[end,end] = 1/h^2 - im * matrix_entry * (1.0/h)
    
#     return Lap1D
# end

# function getLaplace2DMatrix(n::Vector{Int64}, h::Vector{Float64}, matrix)
#     In(n::Int) = spdiagm(0=>ones(ComplexF64, n));
    
#     n_y, n_x = n
#     h_y, h_x = h
#     Lap1D_x = getLaplace1D(n_x, h_x, sqrt(real(matrix[1,1])))
#     Lap1D_y = getLaplace1D(n_y, h_y, sqrt(real(matrix[end,end])))
    
#     # println(size(kron(In(n_y),Lap1D_x)))
#     # println(size(kron(Lap1D_y,In(n_x))))
#     # println(size(spdiagm(0=>vec(matrix))))
#     Lap2D = kron(In(n_y),Lap1D_x) + kron(Lap1D_y,In(n_x)) - spdiagm(0=>vec(matrix))  # D_xx ⊗ I + I ⊗ D_yy

#     return Lap2D
# end

