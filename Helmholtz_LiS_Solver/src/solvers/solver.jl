using KrylovMethods

include("multigrid_solver.jl")
include("LiS_solver.jl")

function solveLinearSystem(solver::LiS_solver, x, b, n)
    return single_LiS_solve(solver, b) 
end

function solveLinearSystem(solver::MG_solver, x, b, n)
    return MG_solve(solver, x, b, n)
end

# function solve(At, solver::Union{LiS_solver,MG_solver}, n, b, h, helmholtz_matrix, laplacian_stencil, mass_stencil, restart, max_iter)
function solve(solver::Union{LiS_solver,MG_solver}, n, b, h, helmholtz_matrix, sommerfeld, laplacian_stencil, mass_stencil, restart, max_iter)
    x_init = zeros(ComplexF64, size(b))
    b = reshape(b, size(b)..., 1, 1)
    A = v->vec(HelmholtzOperator(reshape(v, size(b)), helmholtz_matrix, sommerfeld, h, laplacian_stencil, mass_stencil))
    M = v->vec(solveLinearSystem(solver, zeros(ComplexF64, size(b)), reshape(v, size(b)), n))  
    
    
    # Az = zeros(eltype(b),size(b)[1:2]);
    # Afun = z->(SpMatMul(At,z,Az,4);return Az;);
    # Afun = z->At'*z
    
    println("GOREN GOREN GOREN")
    x,flag,err,iter,resvec = fgmres(A, vec(b), restart, tol=1e-6, maxIter=max_iter, M=M, x=vec(x_init), out=2, flexible=true)

    return x, length(resvec), err  
end