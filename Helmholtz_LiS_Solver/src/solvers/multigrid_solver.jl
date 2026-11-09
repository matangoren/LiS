using KrylovMethods
using ArrayPadding

include("../operators.jl")
include("../utils.jl")



mutable struct MG_solver
    Rs_Grid
    Rs
    Ps
    laplacian_stencils
    mass_stencils
    alphas
    h
    m
    gamma
    omega
    level::Int64
    relax_iter::Vector{Int64}
    coarse_LiS_solver
    gmres_maxIter::Int64
    gmres_restart::Int64
    cycle::Int64
    orderNeumannBC::Int64
    add_sommerfeld::Bool
end


function getMG_solver(Rs_Grid, Rs, Ps, laplacian_stencils, mass_stencils, alphas, h, m, gamma, omega, level, relax_iter; coarse_LiS_solver=nothing, gmres_maxIter=1, gmres_restart=10, cycle=1,orderNeumannBC=1,add_sommerfeld=true)
    return MG_solver(Rs_Grid, Rs, Ps, laplacian_stencils, mass_stencils, alphas, h, m, gamma, omega, level, relax_iter, coarse_LiS_solver, gmres_maxIter, gmres_restart, cycle,orderNeumannBC,add_sommerfeld)
end


function getJacobiDiagonal(h, matrix, correction, laplacian_stencil, mass_stencil)
    lap_mid = div(size(laplacian_stencil,1),2)+1
    mass_mid = div(size(mass_stencil,1),2)+1
    if length(h) == 2
        D = laplacian_stencil[lap_mid, lap_mid]/(h[1]^2) .+ mass_stencil[mass_mid,mass_mid].*matrix #(-matrix .+ correction[1])
    else
        D = laplacian_stencil[lap_mid, lap_mid,lap_mid]/(h[1]^2) .+ mass_stencil[mass_mid,mass_mid,mass_mid].*matrix #(-matrix .+ correction[1])
    end

    D .+= correction[1] .+ correction[2].*matrix
    # D = ones(size(D)) * mean(D)
    
    return D
end

function helmholtzJacobi(x, b, h, matrix, correction, laplacian_stencil, mass_stencil; w=0.8, max_iter=1)
    lap_mid = div(size(laplacian_stencil,1),2)+1
    mass_mid = div(size(mass_stencil,1),2)+1
    if length(h) == 2
        D = laplacian_stencil[lap_mid, lap_mid]/(h[1]^2) .+ mass_stencil[mass_mid,mass_mid].*matrix #(-matrix .+ correction[1])
    else
        D = laplacian_stencil[lap_mid, lap_mid,lap_mid]/(h[1]^2) .+ mass_stencil[mass_mid,mass_mid,mass_mid].*matrix #(-matrix .+ correction[1])
    end

    D .+= correction[1] .+ correction[2].*matrix #(-matrix .+ correction[1])
    # D = ones(size(D)) * mean(D)

    w_D_inv = w ./ D
    for _ in 1:max_iter
        residual = b .- HelmholtzOperator(x, matrix, correction, h, laplacian_stencil, mass_stencil)
        # if norm(x) > 0
        #     residual -= HelmholtzOperator(x, matrix, correction, h, laplacian_stencil, mass_stencil)
        # end  
        x .+= (w_D_inv.*residual)
    end
    return x
end

function applyJacobi(r, h, matrix, correction, laplacian_stencil, mass_stencil; w=0.8, max_iter=1)
    lap_mid = div(size(laplacian_stencil,1),2)+1
    mass_mid = div(size(mass_stencil,1),2)+1
    if length(h) == 2
        D = laplacian_stencil[lap_mid, lap_mid]/(h[1]^2) .+ mass_stencil[mass_mid,mass_mid].*matrix #(-matrix .+ correction[1])
    else
        D = laplacian_stencil[lap_mid, lap_mid,lap_mid]/(h[1]^2) .+ mass_stencil[mass_mid,mass_mid,lap_mid].*matrix #(-matrix .+ correction[1])
    end

    D .+= correction[1] .+ correction[2].*matrix #(-matrix .+ correction[1])

    # D = ones(size(D)) * mean(D)

    # N = prod(size(matrix))
    # A = zeros(ComplexF64, N, N)

    # for j in 1:N
    #     e = zeros(ComplexF64, size(matrix)...,1,1)
    #     e[j] = 1
    #     A[:, j] .= vec(HelmholtzOperator(
    #         e, matrix, correction, h,
    #         laplacian_stencil, mass_stencil
    #     ))
    # end

    # D_true = diag(A)
    # D_constructed = vec(D)
    # relative_error = norm(D_true - D_constructed) / norm(D_true)
    # @show relative_error

    w_D_inv = w ./ D

    return w_D_inv.*r
end

function smoothSPAI(x, b, h, matrix, correction, laplacian_stencil, mass_stencil, Q; w=0.8, max_iter=1)
    for _ in 1:max_iter
        residual = b .- HelmholtzOperator(x, matrix, correction, h, laplacian_stencil, mass_stencil)
        x .+= w.*Q.*residual
    end
    return x
end



# cycle - multigrid cycle type - 1 for V-cycle and 2 for W-cycle
# matrices is a list of matrics (sl_m, helmholtz_m, correction) per level. Constructed one time at solver setup.
function helmholtzVCycle(n, x, b, h, matrices, Rs, Ps, laplacian_stencils, mass_stencils, alphas,ws; level=3, relax_iter=[1,1], coarse_LiS_solver=nothing, maxIter=1, restart=10,cycle=1,coarse_tol=0.1)
    sl_m, helmholtz_m, correction, Q = matrices[level]
    w = ws[level]
    
    if level > 2
        R = Rs[level-1]
        P = Ps[level-1]
        n_coarse = div.(n,2)
        x = helmholtzJacobi(x, b, h, sl_m, correction, laplacian_stencils[level], mass_stencils[level]; max_iter=relax_iter[1],w=w)
        # x = smoothSPAI(x, b, h, sl_m, correction, laplacian_stencils[level], mass_stencils[level],Q; max_iter=relax_iter[1],w=w)

        r = b - HelmholtzOperator(x, sl_m,correction, h, laplacian_stencils[level], mass_stencils[level])          
        r_coarse = R(real(r)) .+ im*R(imag(r))
        e_coarse = similar(r_coarse,ComplexF64,(n_coarse .+ 1)...,1,1)
        fill!(e_coarse, 0)
        for i=1:cycle
            e_coarse = helmholtzVCycle(n_coarse, e_coarse, r_coarse, h.*2, matrices, Rs, Ps, laplacian_stencils, mass_stencils, alphas, ws;
                                level=level-1, relax_iter=relax_iter, coarse_LiS_solver=coarse_LiS_solver, maxIter=maxIter, restart=restart,cycle=cycle,coarse_tol=coarse_tol)
            
        end
        fine_error = (P(real(e_coarse)) .+ im * P(imag(e_coarse)))
        x .+= fine_error
        x = helmholtzJacobi(x, b, h, sl_m, correction, laplacian_stencils[level], mass_stencils[level]; max_iter=relax_iter[2],w=w)
        # x = smoothSPAI(x, b, h, sl_m, correction, laplacian_stencils[level], mass_stencils[level],Q; max_iter=relax_iter[1],w=w)
        
    elseif level == 2
        R = Rs[level-1]
        P = Ps[level-1]
        n_coarse = div.(n,2)
        x = helmholtzJacobi(x, b, h, sl_m, correction, laplacian_stencils[level], mass_stencils[level]; max_iter=relax_iter[1],w=w)
        # x = smoothSPAI(x, b, h, sl_m, correction, laplacian_stencils[level], mass_stencils[level],Q; max_iter=relax_iter[1],w=w)

        r = b - HelmholtzOperator(x, sl_m,correction, h, laplacian_stencils[level], mass_stencils[level])          
        r_coarse = R(real(r)) + im*R(imag(r))
        e_coarse = similar(r_coarse,ComplexF64,(n_coarse .+ 1)...,1,1)
        fill!(e_coarse, 0)
        for i=1:1
            e_coarse = helmholtzVCycle(n_coarse, e_coarse, r_coarse, h.*2, matrices, Rs, Ps, laplacian_stencils, mass_stencils, alphas, ws;
                                level=level-1, relax_iter=relax_iter, coarse_LiS_solver=coarse_LiS_solver, maxIter=maxIter, restart=restart,cycle=cycle,coarse_tol=coarse_tol)
            
        end
        fine_error = (P(real(e_coarse)) + im * P(imag(e_coarse)))
        x .+= fine_error
        x = helmholtzJacobi(x, b, h, sl_m, correction, laplacian_stencils[level], mass_stencils[level]; max_iter=relax_iter[2],w=w)
        # x = smoothSPAI(x, b, h, sl_m, correction, laplacian_stencils[level], mass_stencils[level],Q; max_iter=relax_iter[1],w=w)

    else
        x_size = size(x)
       
        if coarse_LiS_solver == nothing
            A_Coarsest(v) =  vec(HelmholtzOperator(reshape(v,x_size), sl_m, correction, h, laplacian_stencils[level], mass_stencils[level]))
            # M_Coarsest(v) = vec(applyJacobi(reshape(v,x_size), h, sl_m, correction, laplacian_stencils[level], mass_stencils[level]; max_iter=1, w=w))
            # M_Coarsest(v) = vec(w.*Q.*reshape(v,x_size))
            if device === gpu
                CUDA.synchronize()
            end
            t0 = time_ns()
            x,flag,err,iter,resvec = fgmres_func(A_Coarsest, vec(b), restart, tol=coarse_tol, maxIter=maxIter, x=vec(x), out=-1, flexible=true)
            # x,flag,err,iter,resvec = fgmres_func(A_Coarsest, vec(b), restart, tol=coarse_tol, maxIter=maxIter,M=M_Coarsest, x=vec(x), out=-1, flexible=true)
            if device === gpu
                CUDA.synchronize()
            end
            t = (time_ns() - t0) / 1e9   # seconds
            push!(coarse_times, t)
            push!(coarse_iter, length(resvec))
            push!(coarse_err, err)
        else
            A_Coarsest_LiS(v) =  vec(HelmholtzOperator(reshape(v,x_size), sl_m, correction, h, laplacian_stencils[level], mass_stencils[level]))
            # M_Coarsest_LiS(v) = vec(coarse_LiS_solver.T_Jacobi.*helmholtzJacobi(x, reshape(v,x_size), h, sl_m, correction, laplacian_stencils[level], mass_stencils[level]; max_iter=1, w=w) + weighted_LiS_solve(coarse_LiS_solver, reshape(v,x_size)))
            M_Coarsest_LiS(v) = vec(weighted_LiS_solve(coarse_LiS_solver, reshape(v,x_size)))
            # M_Coarsest_LiS(v) = vec(coarse_LiS_solver.T_SPAI.*coarse_LiS_solver.Q_SPAI.*reshape(v,x_size) .+ weighted_LiS_solve(coarse_LiS_solver, reshape(v,x_size)))
            
            if device === gpu
                CUDA.synchronize()
            end
            t0 = time_ns()
            x,flag,err,iter,resvec = fgmres_func(A_Coarsest_LiS, vec(b), restart, tol=coarse_tol, maxIter=maxIter, M=M_Coarsest_LiS, x=vec(x), out=-1, flexible=true)
            if device === gpu
                CUDA.synchronize()
            end
            t = (time_ns() - t0) / 1e9   # seconds
            push!(coarse_times, t)
            push!(coarse_iter, length(resvec))
            push!(coarse_err, err)
            # x = M_Coarsest_LiS(b)
        end
        x = reshape(x, x_size)
    end

    return x
end

laplacian_kernel = zeros(data_type, 3, 3, 3)
laplacian_kernel[2,2,2] = 6
laplacian_kernel[1,2,2] = -1
laplacian_kernel[3,2,2] = -1
laplacian_kernel[2,1,2] = -1
laplacian_kernel[2,3,2] = -1
laplacian_kernel[2,2,1] = -1
laplacian_kernel[2,2,3] = -1

mass_kernel = zeros(data_type, 3, 3, 3)
mass_kernel[2,2,2] = 1

high_laplacian_3d = [0.0;1;0;;1;2;1;;0;1;0;;;1;2;1;;2;-24;2;;1;2;1;;;0;1;0;;1;2;1;;0;1;0]*(-1/6)

high_mass_3d = zeros(data_type, 3, 3, 3)
high_mass_3d[1,2,2] = 1
high_mass_3d[2,:,:] = [0 1 0; 1 6 1; 0 1 0]
high_mass_3d[3,2,2] = 1

high_mass_3d = high_mass_3d.*(1/12)



laplacian_types_3D = Dict("Low" => reshape(laplacian_kernel, size(laplacian_kernel)...,1,1), "High" => reshape(high_laplacian_3d, size(high_laplacian_3d)...,1,1))
mass_types_3D = Dict("Low" => reshape(mass_kernel, size(mass_kernel)...,1,1), "High" => reshape(high_mass_3d, size(high_mass_3d)...,1,1))

laplacian_types_2D = Dict("Low" => reshape([0.0 -1.0 0.0; -1.0 4.0 -1.0; 0.0 -1.0 0.0],3,3,1,1), "High" => reshape([-1/6 -2/3 -1/6; -2/3 10/3 -2/3; -1/6 -2/3 -1/6],3,3,1,1));
mass_types_2D = Dict("Low" => reshape([0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0],3,3,1,1), "High" => reshape([0 1/12 0; 1/12 2/3 1/12; 0 1/12 0],3,3,1,1));

Interpolation_types = Dict("Low" => 0.5 * [1,2,1], "High" => (1/8) * [1,4,6,4,1]);

function getInterp(s; dim=2)
    P = copy(s)
    println(dim)
    k = Int64.(length(s) .* ones(dim))
    for i=2:dim
        P = kron(s,P)
    end
    P = reshape(P, k...,1,1)
    R = P ./ 2^dim

    return R,P
end


function getCoarseStencils(R, P, laplacian_stencil, mass_stencil,level; dim=2)
    s = size(P,1)
    # coarse_laplacian_stencil
    x = zeros((s+2)*ones(Int64,dim)...,1,1)
    if dim == 2
        x[div(s+2,2)+1, div(s+2,2)+1,1,1] = 1.0
    elseif dim == 3
        x[div(s+2,2)+1, div(s+2,2)+1, div(s+2,2)+1,1,1] = 1.0
    else
        throw("dim must be 2 or 3")
    end
    x = ConvTranspose(P, zeros(data_type, 1), stride=2,pad=div(s,2))(x)
    x = Conv(laplacian_stencil, zeros(data_type, 1),stride=1, pad=div(size(laplacian_stencil,1),2))(x);
    x = Conv(R, zeros(data_type, 1),stride=2, pad=div(s,2))(x)

    if level < 3 || s <= 3
        if dim == 2
            x = x[2:end-1,2:end-1,:,:]
        else
            x = x[2:end-1,2:end-1,2:end-1,:,:]
        end
    end
    coarse_laplacian_stencil = copy(x)*4

    # coarse_mass_stencil
    x = zeros((s+2)*ones(Int64,dim)...,1,1)
    if dim == 2
        x[div(s+2,2)+1, div(s+2,2)+1,1,1] = 1.0
    elseif dim == 3
        x[div(s+2,2)+1, div(s+2,2)+1, div(s+2,2)+1,1,1] = 1.0
    else
        throw("dim must be 2 or 3")
    end
    x = ConvTranspose(P, zeros(data_type, 1), stride=2,pad=div(s,2))(x)
    x = Conv(mass_stencil, zeros(data_type, 1),stride=1, pad=div(size(mass_stencil,1),2))(x);
    x = Conv(R, zeros(data_type, 1),stride=2, pad=div(s,2))(x)
    
    if level < 3 || s <= 3
        if dim == 2
            x = x[2:end-1,2:end-1,:,:]
        else
            x = x[2:end-1,2:end-1,2:end-1,:,:]
        end
    end
    coarse_mass_stencil = copy(x)

    return coarse_laplacian_stencil, coarse_mass_stencil
end


function buildCycle(level, P_types, fine_laplacian_type::String, fine_mass_type::String; dim=2)
    if dim == 2
        laplacian_types = laplacian_types_2D
        mass_types = mass_types_2D
    else
        laplacian_types = laplacian_types_3D
        mass_types = mass_types_3D
    end
    laplacian_stencils = Vector{}(undef, level)
    mass_stencils = Vector{}(undef, level)
    Ps = Vector{}(undef, level-1)
    Rs = Vector{}(undef, level-1)
    Rs_Grid = Vector{}(undef, level-1)

    
    laplacian_stencils[level] = copy(laplacian_types[fine_laplacian_type])
    mass_stencils[level] = copy(mass_types[fine_mass_type])

    for i=(level-1):-1:1
        interpolation_1d_stencil = Interpolation_types[P_types[i]]
        interp_length = length(interpolation_1d_stencil)
        pad = div(interp_length,2)
        R,P = getInterp(interpolation_1d_stencil; dim=dim)
        Rs[i] = Conv(R, zeros(data_type,1), stride=2,pad=pad)
        
        Rs_Grid[i] = Rs[i]
        # for no border artifacts (zeros)
        # pad_r = Tuple(ones(Int64,dim*2)*pad)
        # Rs_Grid[i] = x->Conv(R, zeros(Float64,1), stride=2,pad=0)(pad_repeat(x,pad_r)) # for low order this is better
        
        # if interp_length == 3
        #     Ps[i] = ConvTranspose(P, zeros(Float64, 1), stride=2,pad=pad)
        # else
        #     Ps[i] = x->ConvTranspose(P, zeros(Float64, 1), stride=2,pad=(interp_length+1))(pad_repeat(x,pad_r))
        # end
        Ps[i] = ConvTranspose(P, zeros(data_type, 1), stride=2,pad=pad)
        

        laplacian_stencils[i], mass_stencils[i]  = getCoarseStencils(R,P, laplacian_stencils[i+1], mass_stencils[i+1], level-i+1 ; dim=dim)
    end

    return Rs_Grid, Rs, Ps, laplacian_stencils, mass_stencils
end