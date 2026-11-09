using FFTW

include("utils.jl")

function getFFTGreensFunction(n::Vector{Int64}, h::Vector{Float64}, m_0::ComplexF64, q::Array{ComplexF64}, pad::Vector{Int64})
    """
    n - size of the grid [n1, n2]
    kernel - the differential operator kernel
    kappa_0 - const value of the incident wavefield slowness sqaured medium
    q - delta function source term
    """
    h1 = -1 / (h[1]^2)
    h2 = -1 / (h[2]^2)
    kernel = ComplexF64.([[0 h1 0];[h2 -2*(h1+h2)-m_0 h2];[0 h1 0]])

    n_padded = n + 2 .* pad
    kernal_op = getKernelOperator(kernel, n_padded)
    q_padded = zeros(ComplexF64,n_padded...)
    q_padded[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2]] .= q

    g = ifft(fft(q_padded) ./ fft(kernal_op))
    
    g = g[div(n[1],2):div(n[1],2)+2*n[1]-1,div(n[2],2):div(n[2],2)+2*n[2]-1] # crop
    
    g = fftshift(g)
    # figure()
    # imshow(real(fft(g))); colorbar();
    
    return fft(g)
end

mutable struct LiS_solver
    Fg::Vector{Array{ComplexF64}}           # F(g) where g is the Green's function of A(κ_0)^-1
    Omega_2::Vector{Array{ComplexF64}}
    pad:: Vector{Int64}                     # Green's padding
    n::Vector{Int64}                        # grid size
    δ::Array{ComplexF64}                    # delta function source term
    h::Vector{Float64}                      # discretization step size
    index::Int64                            # current Fg_inv index
    helmholtz_m
end

function getLiS_solver(n::Vector{Int}, h::Vector{Float64}, δ::Array{ComplexF64}, m0_s::Vector{ComplexF64}, helmholtz_m::Array{ComplexF64}, pad::Vector{Int64})
    """
    n - grid size
    h - grid cell size
    δ - source term for the Green's function
    m0_s - vector of the constant helmholtz matrices used for creating the Green's functions
    helmholtz_m - the original problem helmholtz matrix
    pad - padding of the Green's function
    """
    Fg = Array{ComplexF64}[]
    Omega_2 = Array{ComplexF64}[]
    j = 8
    for m0 in m0_s
        append!(Fg, [getFFTGreensFunction(n, h, m0, δ, pad)])
        etas = ComplexF64.(sign.(randn(prod(n), j)))
        R = secondOrderHelmholtz(reshape(etas, n..., 1, j), helmholtz_m, h)
        R = reshape(applyLiS(Fg[end], reshape(R, n..., j), n, j), size(etas))
        append!(Omega_2, [sum(R .* conj(etas), dims=2) ./ sum(conj(R) .* R, dims=2)])
    end

    return LiS_solver(Fg, Omega_2, pad, n, δ, h, 1, helmholtz_m)
end

function applyLiS(g, b, n, j)
    """
    g - Green's function
    b - rhs batch of vectors of size (n...,j)
    n - grid size
    j - batch size
    """
    b_padded = zeros(ComplexF64,(2 .* n)...,j)
    b_padded[div(n[1],2)+1:div(n[1],2)+n[1],div(n[2],2)+1:div(n[2],2)+n[2],:] .= b
    x = ifft(g .* fft(b_padded, [1,2]), [1,2])

    return x[div(n[1],2)+1:div(n[1],2)+n[1],div(n[2],2)+1:div(n[2],2)+n[2], :]
end

function LiS_solve(solver::LiS_solver, r)
    n = solver.n
    r_padded = zeros(ComplexF64,(2 .* n)...)
    r_padded[div(n[1],2)+1:div(n[1],2)+n[1],div(n[2],2)+1:div(n[2],2)+n[2]] .= r
    e = ifft(solver.Fg[solver.index] .* fft(r_padded))

    return e[div(n[1],2)+1:div(n[1],2)+n[1],div(n[2],2)+1:div(n[2],2)+n[2]]
end
