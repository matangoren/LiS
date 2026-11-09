using FFTW
using SparseArrays
using LinearAlgebra
using Statistics
using PyPlot
using Clustering


export LippmannSolver,getLippmannSolver

include("../solvers/LiS_solver.jl")
include("../solvers/multigrid_solver.jl")
include("../utils.jl")

function kmeans_pixel_centers(A, K)
    x = Float64.(vec(A))
    data = reshape(x, 1, :)
    result = kmeans(data, K)
    centers = vec(result.centers)

    return centers
end


mutable struct ConvGeometricMultigridSolver <: AbstractSolver
    Rs_Grid
    Rs
    Ps
    laplacian_stencils
    mass_stencils
    alphas
    ws
    h
    m
    matrices
    gamma
    omega
    level::Int64
    relax_iter
    coarse_LiS_solver
    gmres_maxIter::Int64
    gmres_restart::Int64
    cycle::Int64
    orderNeumannBC::Int64
    add_sommerfeld::Bool
    coarse_solver_type::String
    coarse_tol
end

function to_gpu(solver::ConvGeometricMultigridSolver)

    # Stencils
    solver.laplacian_stencils = device.(solver.laplacian_stencils)

    solver.mass_stencils = device.(solver.mass_stencils)

    # Restriction / prolongation operators
    solver.Rs = Flux.gpu.(solver.Rs)
    solver.Ps = Flux.gpu.(solver.Ps)
    # Rs_Grid refers to the restriction operators
    solver.Rs_Grid = solver.Rs

    solver.m = device(solver.m)
    solver.gamma = device(solver.gamma)

    solver.matrices = [(device(M[1]),device(M[2]),device.(M[3])) for M in solver.matrices]

    if solver.coarse_LiS_solver !== nothing
        solver.coarse_LiS_solver = to_gpu(solver.coarse_LiS_solver)
    end

    return solver
end

function getConvGeometricMultigridSolver(;laplace_type="Low",mass_type="Low",level=2,coarse_LiS_solver=nothing,gmres_maxIter=20,gmres_restart=10,cycle=2,dim=2,interType="Low",relax_iter=[1,1],orderNeumannBC=1,add_sommerfeld=true, coarse_solver_type="LiS", coarse_tol=0.1)
    # coarse_solver: "gmres" or "LiS"
    Rs_Grid, Rs, Ps, laplacian_stencils, mass_stencils = buildCycle(level, repeat([interType],level-1), laplace_type, mass_type; dim=dim)
    shift = 0.0
    if laplace_type == "Low"
        if level == 3
            shift = 0.3
        elseif level == 4
            shift = 0.5
        end
    else
        if level == 3
            shift = 0.1
        elseif level == 4
            shift = 0.3
        end
    end
    alphas = ones(data_type,level) * shift

    if T == "Low" || dim == 2
        ws = ones(data_type,level)*0.8
    else
        if level == 2
            ws = [0.4,0.6]
        else
            ws = [0.4,0.4,0.6]
        end
    end
    println("****************************************************")
    println("levels: $(level)")
    println("Laplacian order: $(laplace_type), Mass order: $(mass_type)")
    println("Intergrid order: $(interType)")
    println("shift: $(shift)")
    println("relax iterations: $(relax_iter)")
    println("w: $(ws)")
    println("coarse solver: $(coarse_solver_type) - maxIter - $(gmres_maxIter) restart - $(gmres_restart)")
    println("****************************************************")

    return ConvGeometricMultigridSolver(Rs_Grid,Rs,Ps,laplacian_stencils,mass_stencils,alphas,ws,[],[],[],[],0.0,level,relax_iter,coarse_LiS_solver,gmres_maxIter,gmres_restart,cycle,orderNeumannBC,add_sommerfeld,coarse_solver_type,coarse_tol)
end

import Base.isempty
function isempty(solver::ConvGeometricMultigridSolver)
	return isempty(p.m);
end

import jInv.LinearSolvers.copySolver;
function copySolver(solver::ConvGeometricMultigridSolver)
	ConvGeometricMultigridSolver(solver.Rs_Grid,solver.Rs,solver.Ps,solver.laplacian_stencils,solver.mass_stencils,solver.alphas,solver.ws,solver.h,solver.m,solver.matrices,solver.gamma,solver.omega,solver.level,solver.relax_iter,solver.coarse_LiS_solver,solver.gmres_maxIter,solver.gmres_restart,solver.cycle,solver.orderNeumannBC,solver.add_sommerfeld,solver.coarse_solver_type,solver.coarse_tol);
end

import jInv.LinearSolvers.solveLinearSystem!;
function solveLinearSystem!(A,r,X,solver::ConvGeometricMultigridSolver,doTranspose::Int=0)
    n = collect(size(solver.m))

    r = device(r)
    X = device(X)
    result = helmholtzVCycle(n, reshape(X,n...,1,1), reshape(r,n...,1,1), solver.h, solver.matrices, solver.Rs, solver.Ps, solver.laplacian_stencils, solver.mass_stencils, solver.alphas,solver.ws;
                            level=solver.level, relax_iter=solver.relax_iter, coarse_LiS_solver=solver.coarse_LiS_solver, maxIter=solver.gmres_maxIter, restart=solver.gmres_restart, cycle=solver.cycle,coarse_tol=solver.coarse_tol)
    
    result = cpu(vec(result))
    return result, solver
end

function SPAI_onlyJacobi(solver, n, h, m0_s_T2, m0_s_T3, X_new)
    K = 100
    X = sign.(randn(n...,K,1,1));

    sl_m, helmholtz_m, correction = solver.matrices[1]

    AX = zeros(ComplexF64, size(X));
    Threads.@threads for i in axes(X,3)
        @views AX[:,:,i] .= HelmholtzOperator(Array(X[:,:,i,:,:]), sl_m,correction, h, solver.laplacian_stencils[1], solver.mass_stencils[1])
    end


    delta = zeros(ComplexF64, n...)
    if dim == 2
        delta[div(n[1],2)+2,div(n[2],2)+2] = 1.0 # check for even/odd numbers
    else
        delta[div(n[1],2)+2,div(n[2],2)+2,div(n[3],2)+2] = 1.0 # check for even/odd numbers
    end

    D = getJacobiDiagonal(h, sl_m, correction, solver.laplacian_stencils[1], solver.mass_stencils[1])
    
    V1 = AX ./ D # Jacobi

    omega_jacobi = zeros(ComplexF64,n...)

    @views for I in CartesianIndices(size(AX)[1:length(n)])
        Ti = hcat(vec(V1[I,:]));
        vi = vec(X[I,:]);
        omega = Ti\vi;
	    omega_jacobi[I] = omega[1];
    end

    # println("##### IN SPAI onlyJACOBI #####")
    # println("on X norm = $(norm(X - omega_jacobi.*V1) / norm(X))")
    AX_new = zeros(ComplexF64, size(X_new));
    Threads.@threads for i in axes(X,3)
        @views AX_new[:,:,i] .= HelmholtzOperator(Array(X_new[:,:,i,:,:]), sl_m,correction, h, solver.laplacian_stencils[1], solver.mass_stencils[1])
    end
    V1 = AX_new ./ D # Jacobi
    # println("on X_new norm = $(norm(X_new - omega_jacobi.*V1) / norm(X_new))")

    # power method
    # x = randn(n...,1,1)
    # x = x / norm(x)
    # for i=1:10
    #     Ax = HelmholtzOperator(x, sl_m,correction, h, solver.laplacian_stencils[1], solver.mass_stencils[1])
    #     v1 = Ax ./ D
    #     x = x - omega_jacobi.*v1
    #     x = x / norm(x)
    # end

    # Ax = HelmholtzOperator(x, sl_m,correction, h, solver.laplacian_stencils[1], solver.mass_stencils[1])
    # v1 = Ax ./ D
    # Mx = x - omega_jacobi.*v1
    
    # x = vec(x)
    # lambda_max = x'*vec(Mx) / (x'*x)
    # println("Large Eigenvalue for SPAI_onlyJacobi is = $(lambda_max)")

    return omega_jacobi

end

function SPAI_optimization(solver, n, h, lis_solver)
    K = 100
    X = sign.(randn(n...,K,1,1));

    sl_m, helmholtz_m, correction, Q = solver.matrices[1]

    AX = zeros(ComplexF64, size(X));
    if length(n) == 2
        Threads.@threads for i in axes(X,3)
            @views AX[:,:,i] .= HelmholtzOperator(Array(X[:,:,i,:,:]), sl_m,correction, h, solver.laplacian_stencils[1], solver.mass_stencils[1])
        end
    else
        Threads.@threads for i in axes(X,3)
            @views AX[:,:,:,i] .= HelmholtzOperator(Array(X[:,:,:,i,:,:]), sl_m,correction, h, solver.laplacian_stencils[1], solver.mass_stencils[1])
        end
    end


    D = getJacobiDiagonal(h, sl_m, correction, solver.laplacian_stencils[1], solver.mass_stencils[1])
    
    V_jacoby = AX ./ D # Jacobi

    V_SPAI = Q .* AX
    if length(n) == 2
        V_LiS = single_LiS_solve(lis_solver, AX[:,:,:,1,1]) # a vector of all lis solutions
    else
        V_LiS = single_LiS_solve(lis_solver, AX[:,:,:,:,1,1]) # a vector of all lis solutions
    end

    omega_jacobi = zeros(ComplexF64,n...)
    omega_SPAI = zeros(ComplexF64,n...)
    omega_lis = [zeros(ComplexF64,n...) for i=1:length(V_LiS)]
    


    @views for I in CartesianIndices(size(AX)[1:length(n)])
        # Ti = hcat(vec(V_jacoby[I,:]), hcat([vec(V_LiS[i][I,:]) for i=1:length(V_LiS)]...))
        # Ti = hcat(vec(V_SPAI[I,:]), hcat([vec(V_LiS[i][I,:]) for i=1:length(V_LiS)]...))
        Ti = hcat(hcat([vec(V_LiS[i][I,:]) for i=1:length(V_LiS)]...))

        
        vi = vec(X[I,:]);
        omega = Ti\vi;
	    # # omega_jacobi[I] = omega[1];
	    # omega_SPAI[I] = omega[1];
        # for i=2:length(omega)
        #     omega_lis[i-1][I] = omega[i]
        # end
        
        for i=1:length(omega)
            omega_lis[i][I] = omega[i]
        end
    end

    # println("##### IN SPAI #####")
    # println("on X norm = $(norm(X - omega_jacobi.*V1 - omega_LiS_T2.*V2 - omega_LiS_T3.*V3 - omega_LiS_T4.*V4 - omega_LiS_T5.*V5 - omega_LiS_T6.*V6 - omega_LiS_T7.*V7) / norm(X))")
    # X_new = sign.(randn(n...,K,1,1));
    # AX_new = zeros(ComplexF64, size(X_new));
    # Threads.@threads for i in axes(X,3)
    #     @views AX_new[:,:,i] .= HelmholtzOperator(Array(X_new[:,:,i,:,:]), sl_m,correction, h, solver.laplacian_stencils[1], solver.mass_stencils[1])
    # end
    # V1 = AX_new ./ D # Jacobi
    # V2 = similar(AX_new)
    # V3 = similar(AX_new)
    # Threads.@threads for i in axes(AX,3)
    #     @views V2[:,:,i] .= LiS_solve(lis_T2, AX_new[:,:,i,:,:])
    #     @views V3[:,:,i] .= LiS_solve(lis_T3, AX_new[:,:,i,:,:])
    # end
    # println("on X_new norm = $(norm(X_new - omega_jacobi.*V1 - omega_LiS_T2.*V2 - omega_LiS_T3.*V3) / norm(X_new))")


    # power method
    # x = randn(n...,1,1)
    # x = x / norm(x)
    # for i=1:10
    #     Ax = HelmholtzOperator(x, sl_m,correction, h, solver.laplacian_stencils[1], solver.mass_stencils[1])
    #     v1 = Ax ./ D
    #     v2 = LiS_solve(lis_T2, Ax)
    #     v3 = LiS_solve(lis_T3, Ax)
    #     v4 = LiS_solve(lis_T4, Ax)
    #     v5 = LiS_solve(lis_T5, Ax)
    #     v6 = LiS_solve(lis_T6, Ax)
    #     v7 = LiS_solve(lis_T7, Ax)
    #     x = x - omega_jacobi.*v1 - omega_LiS_T2.*v2 - omega_LiS_T3.*v3 - omega_LiS_T4.*v4 - omega_LiS_T5.*v5 - omega_LiS_T6.*v6 - omega_LiS_T7.*v7
    #     x = x / norm(x)
    # end

    # Ax = HelmholtzOperator(x, sl_m,correction, h, solver.laplacian_stencils[1], solver.mass_stencils[1])
    # v1 = Ax ./ D
    # v2 = LiS_solve(lis_T2, Ax)
    # v3 = LiS_solve(lis_T3, Ax)
    # v4 = LiS_solve(lis_T4, Ax)
    # v5 = LiS_solve(lis_T5, Ax)
    # v6 = LiS_solve(lis_T6, Ax)
    # v7 = LiS_solve(lis_T7, Ax)
    # Mx = x - omega_jacobi.*v1 - omega_LiS_T2.*v2 - omega_LiS_T3.*v3 - omega_LiS_T4.*v4 - omega_LiS_T5.*v5 - omega_LiS_T6.*v6 - omega_LiS_T7.*v7

    
    # x = vec(x)
    # lambda_max = x'*vec(Mx) / (x'*x)
    # println("Large Eigenvalue for SPAI is = $(lambda_max)")
    omega_jacobi = 0
    omega_SPAI = 0
    return omega_SPAI, omega_lis #cat(omega_lis...; dims=length(n)+1)
end


function SPAI_optimization_forPLOT(solver, n, h, m0_s_T2, m0_s_T3, m0_s_T4)
    X = sign.(randn(n...,prod(n),1,1));

    X = Matrix{Float64}(I, prod(n), prod(n))
    X = reshape(X, (n...,prod(n),1,1))

    sl_m, helmholtz_m, correction = solver.matrices[1]

    AX = zeros(ComplexF64, size(X));
    Threads.@threads for i in axes(X,3)
        @views AX[:,:,i] .= HelmholtzOperator(Array(X[:,:,i,:,:]), sl_m,correction, h, solver.laplacian_stencils[1], solver.mass_stencils[1])
    end


    delta = zeros(ComplexF64, n...)
    if dim == 2
        delta[div(n[1],2)+2,div(n[2],2)+2] = 1.0 # check for even/odd numbers
    else
        delta[div(n[1],2)+2,div(n[2],2)+2,div(n[3],2)+2] = 1.0 # check for even/odd numbers
    end

    D = getJacobiDiagonal(h, sl_m, correction, solver.laplacian_stencils[1], solver.mass_stencils[1])
    
    V1 = AX ./ D # Jacobi
    lis_T2 = getLiS_solver(squeezeConvResult(solver.laplacian_stencils[1]),squeezeConvResult(solver.mass_stencils[1]), n, h, delta, m0_s_T2, div.(n,2), div.(n,2))
    # lis_T3 = getLiS_solver(squeezeConvResult(solver.laplacian_stencils[1]),squeezeConvResult(solver.mass_stencils[1]), n, h, delta, m0_s_T3, div.(n,2), div.(n,2))
    # lis_T4 = getLiS_solver(squeezeConvResult(solver.laplacian_stencils[1]),squeezeConvResult(solver.mass_stencils[1]), n, h, delta, m0_s_T4, div.(n,2), div.(n,2))

    V2 = similar(AX)
    # V3 = similar(AX)
    # V4 = similar(AX)
   
    Threads.@threads for i in axes(AX,3)
        @views V2[:,:,i] .= LiS_solve(lis_T2, AX[:,:,i,:,:])
        # @views V3[:,:,i] .= LiS_solve(lis_T3, AX[:,:,i,:,:])
        # @views V4[:,:,i] .= LiS_solve(lis_T4, AX[:,:,i,:,:])

    end

    omega_jacobi = zeros(ComplexF64,n...)
    omega_LiS_T2 = zeros(ComplexF64,n...)
    # omega_LiS_T3 = zeros(ComplexF64,n...)
    # omega_LiS_T4 = zeros(ComplexF64,n...)


    @views for I in CartesianIndices(size(AX)[1:length(n)])
        Ti = hcat(vec(V1[I,:]), vec(V2[I,:]))#, vec(V3[I,:]), vec(V4[I,:]));
        vi = vec(X[I,:]);
        omega = Ti\vi;
	    omega_jacobi[I] = omega[1];
        omega_LiS_T2[I] = omega[2];
        # omega_LiS_T3[I] = omega[3];
        # omega_LiS_T4[I] = omega[4];
    end

    M = X - omega_jacobi.*V1 - omega_LiS_T2.*V2 #- omega_LiS_T3.*V3 - omega_LiS_T4.*V4
    println("$(size(M))")
    M = reshape(M,(prod(n),prod(n)))
    println("$(size(M))")

    λ = eigvals(M)

    # --------------------------------------------------
    # Plot eigenvalues in the complex plane
    # --------------------------------------------------
    figure(figsize=(7, 7))

    # Eigenvalues
    scatter(
        real.(λ),
        imag.(λ),
        s=30,
        label="Eigenvalues"
    )

    # Unit circle
    θ = range(0, 2π, length=500)

    plot(
        cos.(θ),
        sin.(θ),
        linewidth=2,
        label="Unit circle"
    )

    # Axes and formatting
    axhline(0, linewidth=0.5)
    axvline(0, linewidth=0.5)

    axis("equal")
    xlabel("Real")
    ylabel("Imaginary")
    title("Eigenvalues of A")
    grid(true)
    legend()

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------
    folder = joinpath(@__DIR__,"../../test/dump/Eigenvalues")
    mkpath(folder)

    savefig(
        joinpath(folder, "eigenvalues_1LiS_mean.png"),
        dpi=300,
        bbox_inches="tight"
    )

    close()

    
    return omega_jacobi,omega_LiS_T2,omega_LiS_T3, omega_LiS_T4, X_new
    # return omega_jacobi,omega_LiS_T2,omega_LiS_T3, omega_LiS_T4, omega_LiS_T5, omega_LiS_T6, omega_LiS_T7, X_new
end


import jInv.LinearSolvers.setupSolver;
function setupSolver(Hparam::HelmholtzParam, solver::ConvGeometricMultigridSolver)   
    M = Hparam.Mesh
    n = M.n
    h = M.h
    m = reshape(Hparam.m,(n.+1)...)


    omega = Hparam.omega

    dim = length(n)

    solver.h = h
    solver.m = m
    solver.omega = omega
    solver.gamma = reshape(Hparam.gamma,(n.+1)...)
    solver.add_sommerfeld = Hparam.Sommerfeld

    solver.matrices = getMatrices(solver.Rs_Grid, solver.m, solver.omega, solver.gamma, solver.h,solver.laplacian_stencils, solver.mass_stencils; level=solver.level, alphas=solver.alphas, add_sommerfeld=solver.add_sommerfeld, BC=solver.orderNeumannBC)

    m_coarse = get_coarse_m(m,solver.level, solver.Rs_Grid)
    h_coarse = h .* (2^(solver.level-1))
    n_coarse = collect(size(m_coarse))
    
    # define Green's function parameters on coarse grid
    # shift = (maximum(m) - minimum(m)) < 0.1 ? 0.3 : max(((1-mean(m))/(maximum(m) - minimum(m))^2), solver.alphas[1])
    shift = solver.alphas[1]

    # save_submatrix(m_coarse, solver.matrices[1]) # save coarse info for experiments.


    # m0_s_T2 = (omega)^2 .* quantile(vec(m_coarse), [0.25,0.5,0.75]) .* (1 - im*(gamma_0+shift))
    # values = collect((minimum(m_coarse)/maximum(m_coarse)):0.1:1) .* maximum(m_coarse)
    # m0_s_T2 = (omega)^2 .* values.* (1 - im*(gamma_0+shift))
    
    # centers = kmeans_pixel_centers(m_coarse ./ maximum(m_coarse), 4) .* maximum(m_coarse)
    
    centers = kmeans1d_exact(Float64.(vec(m_coarse ./ maximum(m_coarse))), 4).centers .* maximum(m_coarse)
    

    m0_s_T2 = (omega)^2 .* centers .* (1 - im*(gamma_0 + shift))

    delta_coarse = zeros(ComplexF64, n_coarse...)
    if dim == 2
        delta_coarse[div(n_coarse[1],2)+2,div(n_coarse[2],2)+2] = 1.0 # check for even/odd numbers
    else
        delta_coarse[div(n_coarse[1],2)+2,div(n_coarse[2],2)+2,div(n_coarse[3],2)+2] = 1.0 # check for even/odd numbers
    end
    if solver.coarse_solver_type == "LiS"
        lis_solver = getLiS_solver(squeezeConvResult(solver.laplacian_stencils[1]),squeezeConvResult(solver.mass_stencils[1]), n_coarse, h_coarse, delta_coarse, m0_s_T2, div.(n_coarse,2),div.(n_coarse,2))
        T_SPAI, T_LiS = SPAI_optimization(solver, n_coarse, h_coarse, lis_solver)
        lis_solver.T_SPAI = T_SPAI
        lis_solver.T_LiS = T_LiS
        solver.coarse_LiS_solver = lis_solver

    elseif solver.coarse_solver_type == "compactLiS"
        lis_solver = getLiS_solver(squeezeConvResult(solver.laplacian_stencils[1]),squeezeConvResult(solver.mass_stencils[1]), n_coarse, h_coarse, delta_coarse, m0_s_T2, 4*ones(Int64,dim),4*ones(Int64,dim))
        T_SPAI, T_LiS = SPAI_optimization(solver, n_coarse, h_coarse, lis_solver)
        
        lis_solver.T_SPAI = T_SPAI
        lis_solver.T_LiS = T_LiS
        solver.coarse_LiS_solver = lis_solver
    else
        solver.coarse_LiS_solver = nothing
    end

    if device == gpu
        solver = to_gpu(solver)
    end

    return solver
end


mutable struct ConvResidualComp <: AbstractResidualComp
    laplacian_conv
    mass_conv
    h
    helmholtz_m_ext
    correction
    shape
    shape_ext
    code
    pad
    idx
end

import Multigrid.DomainDecomposition.setupResComp;
function setupResComp(solver_ext::ConvGeometricMultigridSolver, IIp_ext_shape,IIp_shape, code)
    code = .!code'[:]
    

    sl_m,helmholtz_m_ext, correction = getHelmholtzMatrices(cpu(solver_ext.m), solver_ext.omega, cpu(solver_ext.gamma), solver_ext.h,cpu(solver_ext.laplacian_stencils[end]), cpu(solver_ext.mass_stencils[end]); alpha=solver_ext.alphas[end], add_sommerfeld=solver_ext.add_sommerfeld, BC=solver_ext.orderNeumannBC,code=code)
    


    laplacian_stencil = cpu(solver_ext.laplacian_stencils[end])
    mass_stencil = cpu(solver_ext.mass_stencils[end])
    laplacian_conv = Conv(laplacian_stencil ./ (solver_ext.h[1])^2, zeros(data_type, 1), pad=0);
    mass_conv = Conv(mass_stencil, zeros(data_type, 1), pad=0);
    
    pad = Tuple(code.*div(size(laplacian_stencil,1),2))
    stencils_diff = div(size(laplacian_stencil,1),2) - div(size(mass_stencil,1),2)
    idx = code * stencils_diff

    return ConvResidualComp(laplacian_conv,mass_conv,solver_ext.h,helmholtz_m_ext,correction,IIp_shape,IIp_ext_shape,code,pad,idx)
end

import Multigrid.DomainDecomposition.computeResidual!;
function computeResidual!(resComp::ConvResidualComp, b,x, r)
    # reshape x and b
    x = reshape(x,resComp.shape_ext...,1,1)
    b = reshape(b,resComp.shape...,1,1)
    r = b
    h = resComp.h
    pad = resComp.pad
    idx = resComp.idx

    # assumption that size of laplace stencil >= size of mass stencil
    L = resComp.laplacian_conv(pad_repeat(x,pad))
    M = resComp.mass_conv(pad_repeat(resComp.helmholtz_m_ext.*x,pad))

    if length(resComp.h) == 2
        M =M[1+idx[1]:end-idx[2],1+idx[3]:end-idx[4]]

    else
        M =M[1+idx[1]:end-idx[2],1+idx[3]:end-idx[4],1+idx[5]:end-idx[6]]
    end
    r -= (L + M)
    return r
end