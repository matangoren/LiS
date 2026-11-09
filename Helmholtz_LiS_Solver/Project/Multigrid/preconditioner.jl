using LinearAlgebra
using KrylovMethods

function getHelmholtzMatrices(m, omega, gamma, h; alpha=0.1)

    sommerfeld = zeros(ComplexF64, size(m))
    sommerfeld[1,:]  .+= (-1/h[1]^2) .+ ((1/h[1]) .* im*omega*sqrt.(m[1,:]))
    sommerfeld[end,:]  .+= (-1/h[1]^2) .+ ((1/h[1]) .* im*omega*sqrt.(m[end,:]))
    sommerfeld[:,1]  .+= (-1/h[2]^2) .+  ((1/h[2]) .* im*omega*sqrt.(m[:,1]))
    sommerfeld[:,end] .+= (-1/h[2]^2) .+ ((1/h[2]) .* im*omega*sqrt.(m[:,end]))

    helmholtz_matrix = (omega.^2).*m.*(1.0.-1im.*gamma/omega) .- sommerfeld
    sl_matrix = (omega.^2).*m.*(1.0.-1im.*(gamma/omega .+ alpha)) .- sommerfeld


    return sl_matrix, helmholtz_matrix
end


# x is assumed to be of size (H,W,C,N) where N = batch size
# kappa and gamma are two-dimensional (H,W)
function helmholtzJacobi(x, b, h, matrix, laplacian_stencil, mass_stencil; w=0.8, max_iter=1)
    D = 2.0*(sum(1 ./ h.^2)) .- matrix 
    w_D_inv = w ./ D
    for _ in 1:max_iter
        residual = b - HelmholtzOperator(x, matrix, h, laplacian_stencil, mass_stencil)  
        x += (w_D_inv.*residual)
    end
    return x
end

function getInterp(dim; highOrder=false)
    s = 0.5 * [1,2,1]
    pad = 1
    if highOrder
        s = (1/8) * [1,4,6,4,1]
        pad = 2
    end
    k = Int64.(length(s) .* ones(dim))
    P = copy(s)

    for i=2:dim
        P = kron(s,P)
    end
    P = Float64.(reshape(P,k...,1,1))
    R = P ./ 2^dim

    R = Conv(R, zeros(Float64,1), stride=2,pad=pad)
    P = ConvTranspose(P, zeros(Float64, 1), stride=2,pad=pad)
    return R,P
end


function helmholtzVCycle(n, x, b, h, m, gamma, omega, R, P, laplacian_stencils, mass_stencils, alphas; smoother=helmholtzJacobi, level=3, relax_iter=1, coarse_LiS_solver=nothing)
    sl_m, helmholtz_m = getHelmholtzMatrices(m, omega, gamma, h; alpha=alphas[level])
    x = smoother(x, b, h, sl_m, laplacian_stencils[level], mass_stencils[level]; max_iter=relax_iter)

    if level > 1
        r = b - HelmholtzOperator(x, sl_m, h, laplacian_stencils[level], mass_stencils[level])

        m_coarse = R(reshape(m,size(m)...,1,1))[:,:,1,1]
        # gamma_coarse = restriction(reshape(gamma,size(gamma)...,1,1))[:,:,1,1] 

        r_coarse = R(real(r)) + im*R(imag(r))

        n_coarse = div.(n,2)
        gamma_coarse = ones((n_coarse.+1)...) .* 0.01*omega
        e_coarse = zeros(ComplexF64, (n_coarse.+1)...,1,1)

        e_coarse = helmholtzVCycle(n_coarse, e_coarse, r_coarse, h.*2, m_coarse, gamma_coarse, omega, R, P, laplacian_stencils, mass_stencils, alphas;
                            smoother=smoother, level=level-1, relax_iter=relax_iter, coarse_LiS_solver=coarse_LiS_solver)
        
        fine_error = (P(real(e_coarse)) + im * P(imag(e_coarse)))
        x .+= fine_error

    else
        # coarsest grid
        x_size = size(x)
        if coarse_LiS_solver == nothing
            A_Coarsest(v) =  vec(HelmholtzOperator(reshape(v,x_size), sl_m, h, laplacian_stencils[level], mass_stencils[level]))
            M_Coarsest(v) = vec(smoother(x, reshape(v,x_size), h, sl_m, laplacian_stencils[level], mass_stencils[level]; max_iter=1))
            x, flag, err, iter, resvec = fgmres(A_Coarsest, vec(b), 10, tol=0.01, maxIter=15,
                                                M=M_Coarsest, x=vec(x), out=-1, flexible=true)      
        else
            x = LiS_solve(coarse_LiS_solver, b)
            for i=1:5
                e = LiS_solve(coarse_LiS_solver, b-HelmholtzOperator(reshape(x,x_size...),sl_m, h, laplacian_stencils[level], mass_stencils[level]))
                x = x + e
            end
        end
        x = reshape(x, x_size)
    end

    x = smoother(x, b, h, sl_m, laplacian_stencils[level], mass_stencils[level]; max_iter=relax_iter)
    return x
end


mutable struct MG_solver
    R
    P
    laplacian_stencils
    mass_stencils
    alphas
    smoother
    h::Vector{Float64}
    m::Array{Float64}
    gamma
    omega::Float64
    level::Int64
    relax_iter::Int64
    coarse_LiS_solver
end


function getMG_solver(R, P, laplacian_stencils, mass_stencils, alphas, smoother, h, m, gamma, omega, level, relax_iter; coarse_LiS_solver=nothing)
    return MG_solver(R, P, laplacian_stencils, mass_stencils, alphas, smoother, h, m, gamma, omega, level, relax_iter, coarse_LiS_solver)
end

function MG_solve(solver::MG_solver, x, b, n)
    return helmholtzVCycle(n, x, b, solver.h, solver.m, solver.gamma, solver.omega, solver.R, solver.P, solver.laplacian_stencils, solver.mass_stencils, solver.alphas;
                            smoother=solver.smoother, level=solver.level, relax_iter=solver.relax_iter, coarse_LiS_solver=solver.coarse_LiS_solver)
end