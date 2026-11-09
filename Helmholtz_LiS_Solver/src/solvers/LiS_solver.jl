using FFTW
using SparseArrays
using LinearAlgebra



function getKernelOperator3D(kernel::AbstractArray, n)
    kx, ky, kz = size(kernel)
    sx, sy, sz = div.(size(kernel), 2)
    kernel_op = similar(kernel, ComplexF64, n...)
    fill!(kernel_op, 0)

    # Octant mapping (similar to ifftshift)
    kernel_op[1:kx-sx, 1:ky-sy, 1:kz-sz] = kernel[sx+1:end, sy+1:end, sz+1:end]
    kernel_op[end-sx+1:end, 1:ky-sy, 1:kz-sz] = kernel[1:sx, sy+1:end, sz+1:end]
    kernel_op[1:kx-sx, end-sy+1:end, 1:kz-sz] = kernel[sx+1:end, 1:sy, sz+1:end]
    kernel_op[end-sx+1:end, end-sy+1:end, 1:kz-sz] = kernel[1:sx, 1:sy, sz+1:end]

    kernel_op[1:kx-sx, 1:ky-sy, end-sz+1:end] = kernel[sx+1:end, sy+1:end, 1:sz]
    kernel_op[end-sx+1:end, 1:ky-sy, end-sz+1:end] = kernel[1:sx, sy+1:end, 1:sz]
    kernel_op[1:kx-sx, end-sy+1:end, end-sz+1:end] = kernel[sx+1:end, 1:sy, 1:sz]
    kernel_op[end-sx+1:end, end-sy+1:end, end-sz+1:end] = kernel[1:sx, 1:sy, 1:sz]

    return kernel_op
end



function getKernelOperator(kernel::AbstractArray, n::Vector{Int64})
    k_n = size(kernel, 1)
    s = div(k_n,2)
    kernel_op = similar(kernel, ComplexF64, n...)
    fill!(kernel_op, 0)
    kernel_op[1:k_n-s,1:k_n-s] = kernel[s+1:end,s+1:end]
    kernel_op[end-s+1:end,1:k_n-s] = kernel[1:s,s+1:end]
    kernel_op[1:k_n-s,end-s+1:end] = kernel[s+1:end,1:s]
    kernel_op[end-s+1:end,end-s+1:end] = kernel[1:s,1:s]
    return kernel_op
end

function get3dFFTGreensFunction(kernel::AbstractArray,mass_kernel::AbstractArray,n::Vector{Int64}, h::Vector{Float64}, m_0::ComplexF64, q::AbstractArray{ComplexF64}, solver_pad::Vector{Int64}, green_pad::Vector{Int64})
    kernel = ComplexF64.(kernel ./ (h[1]^2)) 
    k_n = size(kernel,1)
    m_n = size(mass_kernel,1)

    kernel[(div(k_n,2)+1-div(m_n,2)):(div(k_n,2)+1+div(m_n,2)), (div(k_n,2)+1-div(m_n,2)):(div(k_n,2)+1+div(m_n,2)), (div(k_n,2)+1-div(m_n,2)):(div(k_n,2)+1+div(m_n,2))] -= (m_0.*mass_kernel)

    n_padded = n + 2solver_pad + 2green_pad
    kernel_op = getKernelOperator3D(kernel, n_padded)
    q_padded = similar(q, ComplexF64, n_padded...)
    fill!(q_padded, 0)

    x = solver_pad + green_pad 
    q_padded[x[1]+1:x[1]+n[1],x[2]+1:x[2]+n[2],x[3]+1:x[3]+n[3]] .= q
    
    g = ifft(fft(q_padded) ./ fft(kernel_op))
    g = g[green_pad[1]+1:green_pad[1]+(n + 2solver_pad)[1],green_pad[2]+1:green_pad[2]+(n + 2solver_pad)[2],green_pad[3]+1:green_pad[3]+(n + 2solver_pad)[3]] # crop
    g = fftshift(g)    
    
    return fft(g)

end

function getFFTGreensFunction(kernel::AbstractArray,mass_kernel::AbstractArray,n::Vector{Int64}, h::Vector{Float64}, m_0::ComplexF64, q::AbstractArray{ComplexF64}, solver_pad::Vector{Int64}, green_pad::Vector{Int64})
    """
    n - size of the grid [n1, n2]
    kernel - the differential operator kernel
    kappa_0 - const value of the incident wavefield slowness sqaured medium
    q - delta function source term
    """
    if length(n) == 3
        return get3dFFTGreensFunction(kernel,mass_kernel,n,h,m_0,q,solver_pad,green_pad)
    end
    kernel = ComplexF64.(kernel ./ (h[1]^2))
    k_n = size(kernel,1)
    m_n = size(mass_kernel,1)
    
    kernel[(div(k_n,2)+1-div(m_n,2)):(div(k_n,2)+1+div(m_n,2)), (div(k_n,2)+1-div(m_n,2)):(div(k_n,2)+1+div(m_n,2))] -= (m_0.*mass_kernel)
    
    n_padded = n + 2solver_pad + 2green_pad
    kernel_op = getKernelOperator(kernel, n_padded)
    q_padded = similar(q, ComplexF64, n_padded...)
    fill!(q_padded, 0)

    x = solver_pad + green_pad 
    q_padded[x[1]+1:x[1]+n[1],x[2]+1:x[2]+n[2]] .= q
    
    g = ifft(fft(q_padded) ./ fft(kernel_op))
    g = g[green_pad[1]+1:green_pad[1]+(n + 2solver_pad)[1],green_pad[2]+1:green_pad[2]+(n + 2solver_pad)[2]] # crop
    g = fftshift(g)

    
    return fft(g)
end

mutable struct LiS_solver
    Fg                                      # F(g) where g is the Green's function of A(κ_0)^-1
    solver_pad:: Vector{Int64}              # solver padding
    green_pad::Vector{Int64}                # inner Green's padding
    n::Vector{Int64}                        # grid size
    δ                                       # delta function source term
    h::Vector{Float64}                      # discretization step size
    T_SPAI
    Q_SPAI
    T_LiS # cat(solver.T_LiS...; dims=length(n)+1), stacked arrays
end

function getLiS_solver(kernel::AbstractArray,mass_kernel::AbstractArray, n::Vector{Int}, h::Vector{Float64}, δ::AbstractArray, m0_s::Vector{ComplexF64}, solver_pad::Vector{Int64}, green_pad::Vector{Int64})
    """
    n - grid size
    h - grid cell size
    δ - source term for the Green's function
    m0_s - vector of the constant helmholtz matrices used for creating the Green's functions
    pad - padding of the Green's function
    """
    # Fg = cat([getFFTGreensFunction(kernel,mass_kernel,n,h,m0,δ,solver_pad,green_pad) for m0 in m0_s]...; dims=length(n)+1)
    Fg = [getFFTGreensFunction(kernel,mass_kernel,n,h,m0,δ,solver_pad,green_pad) for m0 in m0_s]

    return LiS_solver(Fg, solver_pad, green_pad, n, δ, h, 0,0,zeros(Float64, length(m0_s)))
end




function single_LiS_solve(solver::LiS_solver, B)
    # B: a N1xN2[xN3]xK Array
    # return vector containing [LiS_i(B) for i]
  
    n = solver.n
    pad = solver.solver_pad
    B_padded = similar(B, ComplexF64, (n + 2pad)...,size(B,length(n)+1))
    fill!(B_padded, 0)

    if length(n) == 2
        B_padded[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2],:] .= B
        B_padded_fft = fft(B_padded,(1,2))
    else
        B_padded[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2],pad[3]+1:pad[3]+n[3],:] .= B
        B_padded_fft = fft(B_padded,(1,2,3))
    end
    
    results = Vector{AbstractArray{ComplexF64}}(undef, length(solver.Fg))
    for i=1:length(solver.Fg)
        if length(n) == 2
            results[i] = ifft(solver.Fg[i] .* B_padded_fft, (1,2))[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2],:]
        else
            results[i] = ifft(solver.Fg[i] .* B_padded_fft, (1,2,3))[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2],pad[3]+1:pad[3]+n[3],:]
        end
    end
    
    return results
end

# function weighted_LiS_solve(solver::LiS_solver, r)
#     n = solver.n
#     pad = solver.solver_pad
#     r_padded = similar(r, ComplexF64, (n + 2pad)...)
#     fill!(r_padded, 0)

#     if length(n) == 2
#         r_padded[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2]] .= r
#     else
#         r_padded[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2],pad[3]+1:pad[3]+n[3]] .= r
#     end

#     r_padded_fft = reshape(fft(r_padded), size(r_padded)...,1)
#     V = ifft(solver.Fg .* r_padded_fft, 1:length(n))
    
#     if length(n) == 2
#         V = V[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2],:]
#     else
#         V = V[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2],pad[3]+1:pad[3]+n[3],:]
#     end

#     e = dropdims(
#         sum(solver.T_LiS .* V, dims=length(n)+1),
#         dims=length(n)+1
#     )

#     return e
# end

function weighted_LiS_solve(solver::LiS_solver, r)
    n = solver.n
    pad = solver.solver_pad
    r_padded = similar(r, ComplexF64, (n + 2pad)...)
    fill!(r_padded, 0)

    if length(n) == 2
        r_padded[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2]] .= r
        r_padded_fft = fft(r_padded)
        
        e = similar(r, ComplexF64)
        fill!(e, 0)
        for i = 1:length(solver.T_LiS)
            e .+= solver.T_LiS[i] .* ifft(solver.Fg[i] .* r_padded_fft)[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2]]
        end
        return e
    else
        r_padded[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2],pad[3]+1:pad[3]+n[3]] .= r
        r_padded_fft = fft(r_padded)

        e = similar(r, ComplexF64)
        fill!(e, 0)
        for i = 1:length(solver.T_LiS)
            e .+= solver.T_LiS[i] .* ifft(solver.Fg[i] .* r_padded_fft)[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2],pad[3]+1:pad[3]+n[3]]
        end
        return e
    end
end

function to_gpu(solver::LiS_solver)

    # Green's function FFTs
    solver.Fg = device.(solver.Fg)

    # Correction / delta array
    solver.δ = device(solver.δ)

    # LiS / Jacobi weights
    solver.T_Jacobi = device(solver.T_Jacobi)

    solver.T_LiS = device.(solver.T_LiS)

    return solver
end
