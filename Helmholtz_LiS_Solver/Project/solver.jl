using KrylovMethods

include("LiS/preconditioner.jl")
include("Multigrid/preconditioner.jl")
include("operators.jl")


mutable struct Multi_solver
    solvers::Vector         # vector containing our preconditioners [LiS, MG] (in that order)
    alternate::Bool         # true=alternate between preconditioners, false=one after the other
    index::Int64            # when alternate=true the index indicates the current preconditioner
end

function getMulti_solver(solvers; alternate=false)
    return Multi_solver(solvers, alternate, 1)
end

# solving inside FGMRES - Ae=r
function solveLinearSystem(solver::LiS_solver, x, b, n)
    res = LiS_solve(solver, b)
    solver.index = mod((solver.index),length(solver.Fg))+1
    
    return res
end

function solveLinearSystem(solver::MG_solver, x, b, n)
    return MG_solve(solver, x, b, n)
end

function solveLinearSystem(solver::Multi_solver, x, b, n)
    # if solver.alternate
    #     res = solveLinearSystem(solver.solvers[solver.index], x, b, n)
    #     solver.index = mod((solver.index),length(solver.solvers))+1
    #     return res
    # end

    # e = solveLinearSystem(solver.solvers[1], x, b, n)
    # A = solver.solvers[1].A
    # _, helmholtz_matrix = getHelmholtzMatrices(solver.solvers[1].m, solver.solvers[1].omega, solver.solvers[1].gamma, solver.solvers[1].h; alpha=0.5)

    # return e+solveLinearSystem(solver.solvers[2], x, b-A(e, helmholtz_matrix, solver.solvers[1].h), n)

    # blockFGMRES
    res_LiS = solveLinearSystem(solver.solvers[1], reshape(x[:,:,1,1],n...,1,1), reshape(b[:,:,1,1],n...,1,1), n)
    res_MG = solveLinearSystem(solver.solvers[2], reshape(x[:,:,1,2],n...,1,1), reshape(b[:,:,1,2],n...,1,1), n)
    return cat(res_LiS, res_MG, dims=2)
end

function fgmresWrapper(A, M, b, restart, max_iter; gmres_type="FGMRES")
    x_init = zeros(ComplexF64, size(b))   
    
    if gmres_type == "FGMRES"
    x,flag,err,iter,resvec = fgmres(A, vec(b), restart, tol=1e-6, maxIter=max_iter,
                                                    M=M, x=vec(x_init), out=-1, flexible=true)
    elseif gmres_type == "blockFGMRES"
        # blockFGMRES
        x,flag,err,iter,resvec = blockFGMRES(A, b, restart, tol=1e-6, maxIter=max_iter,
                                                        M=M, X=x_init, out=-1, flexible=true)
    else
        error("Incorrect fgmres_type function")
    end   
    return x, length(resvec), err    
end

function solve(solver::Union{LiS_solver,MG_solver,Multi_solver}, n, b, h, m, gamma, omega, restart, max_iter)
    _, helmholtz_matrix = getHelmholtzMatrices(m, omega, gamma, h)
    b_dim = length(size(b))
    if b_dim == 2
        b = reshape(b, size(b)..., 1, 1)
        A = v->vec(HelmholtzOperator(reshape(v, size(b)), helmholtz_matrix, h, [0 -1 0; -1 4 -1; 0 -1 0], [0 0 0; 0 1 0; 0 0 0]))
        M = v->vec(solveLinearSystem(solver, zeros(ComplexF64, size(b)), reshape(v, size(b)), n))
        return fgmresWrapper(A, M, b, restart, max_iter; gmres_type="FGMRES")
    elseif b_dim == 4
        # with blockGMRES
        A = v->reshape(HelmholtzOperator(reshape(v, size(b)), helmholtz_matrix, h, [0 -1 0; -1 4 -1; 0 -1 0], [0 0 0; 0 1 0; 0 0 0]), prod(n), 2)
        M = v->copy(reshape(solveLinearSystem(solver, zeros(ComplexF64, size(b)), reshape(v, size(b)), n), prod(n), 2))
        return fgmresWrapper(A, M, reshape(b,prod(n),2), restart, max_iter; gmres_type="blockFGMRES")
    else
        error("Incorrect b dimension")
    end                                     
end

