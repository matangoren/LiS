using Flux
using LinearAlgebra
using KrylovMethods
using PyPlot
using Helmholtz
using GLMakie

function plotSlices(v)
    fig = GLMakie.Figure()
    ax = LScene(fig[1, 1], show_axis=true)

    n = size(v)
    x = 1:n[1]
    y = 1:n[2]
    z = n[3]:-1:1

    sgrid = SliderGrid(
        fig[2, 1],
        (label = "yz plane - x axis", range = 1:length(x)),
        (label = "xz plane - y axis", range = 1:length(y)),
        (label = "xy plane - z axis", range = 1:length(z)),
    )

    lo = sgrid.layout
    nc = ncols(lo)

    vol = real(v)
    plt = volumeslices!(ax, x, y, z, vol, colormap=:jet)

    # connect sliders to `volumeslices` update methods
    sl_yz, sl_xz, sl_xy = sgrid.sliders

    on(sl_yz.value) do v; plt[:update_yz][](v) end
    on(sl_xz.value) do v; plt[:update_xz][](v) end
    on(sl_xy.value) do v; plt[:update_xy][](n[3]-v) end

    set_close_to!(sl_yz, .5length(x))
    set_close_to!(sl_xz, .5length(y))
    set_close_to!(sl_xy, .5length(z))

    hmaps = [plt[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
    display(GLMakie.Screen(),fig)
end

function getHelmholtzMatrices(m, omega, gamma, h; alpha=0.2, Sommerfeld=true, NeumannOnTop=false)
    somm = zeros(ComplexF64, size(m))
    if Sommerfeld
        somm[1,:,:]  .+= ((-1/h[1]^2) .+ im*omega*sqrt.(m[1,:,:])./h[1])
        somm[end,:,:]  .+= ((-1/h[1]^2) .+ im*omega*sqrt.(m[end,:,:])./h[1])
        somm[:,1,:]  .+= ((-1/h[2]^2) .+ im*omega*sqrt.(m[:,1,:])./h[2])
        somm[:,end,:] .+= ((-1/h[2]^2) .+ im*omega*sqrt.(m[:,end,:])./h[2])
        if !NeumannOnTop
            somm[:,:,1] += ((-1/h[3]^2) .+ im*omega*sqrt.(m[:,:,1])./h[3])
        end
        somm[:,:,end] .+= ((-1/h[3]^2) .+ im*omega*sqrt.(m[:,:,end])./h[3])
    end
    helmholtz_matrix = (omega.^2).*m.*(1.0.-1im.*gamma./real(omega)) .- somm
    sl_matrix = (omega.^2).*m.*(1.0.-1im.*((gamma)./real(omega) .+ alpha)) .- somm


    return sl_matrix, helmholtz_matrix
end



function getInterp(n; highOrder=false)
    s = 0.5 * [1,2,1]
    pad = 1
    if highOrder
        s = (1/8) * [1,4,6,4,1]
        pad = 2
    end
    k = Int64.(length(s) .* ones(length(n)))
    P = copy(s)

    for i=2:length(n)
        P = kron(s,P)
    end
    P = Float64.(reshape(P,k...,1,1))
    R = P ./ 2^length(n)

    R = Conv(R, zeros(Float64,1), stride=2,pad=pad)
    P = ConvTranspose(P, zeros(Float64, 1), stride=2,pad=pad)
    return R,P
end

function getLapacianConv(h::Vector{Float64}; pad=1)
    I = [0,1,0]
    D = [-1,2,-1]
    if length(h) == 2
        Dx = (1/h[2]^2)*D
        Dy = (1/h[1]^2)*D
        L = reshape(kron(Dx, I) + kron(I,Dy),3,3,1,1)
    else
        Dx = (1/h[1]^2)*D
        Dy = (1/h[2]^2)*D
        Dz = (1/h[3]^2)*D
        L = reshape(kron(kron(Dx,I),I) + kron(kron(I,Dy),I) + kron(I,kron(I,Dz)),3,3,3,1,1)

    end
    return Conv(Float64.(L), zeros(Float64, 1); pad=pad)
end

function secondOrderHelmholtz(x::Array{ComplexF64}, matrix, h::Vector{Float64})
    L = getLapacianConv(h; pad=1)
    return L(x) - x.*matrix
    # L = getLapacianConv(h; pad=0)
    # return L(pad_repeat(x, (1,1,1,1,1,1))) - x.*matrix
end

function helmholtzJacobi(x, b, h, matrix; A=secondOrderHelmholtz, w=0.8, max_iter=1)
    D = 2.0*(sum(1 ./ h.^2)) .- matrix 
    w_Dinv = w ./ D
    for _ in 1:max_iter
        residual = b - A(x, matrix, h)  
        x += (w_Dinv.*residual)
    end
    return x
end

function helmholtzVCycle(n, x, b, h, m, gamma, omega, R, P; A=secondOrderHelmholtz, smoother=helmholtzJacobi, level=3, relax_iter=1, coarse_LiS_solver=nothing)
    sl_m, helmholtz_m = getHelmholtzMatrices(m, omega, gamma, h)
    x = smoother(x, b, h, sl_m; A=A, max_iter=relax_iter)

    if level > 1
        r = b - A(x, sl_m, h)
        m_coarse = R(reshape(m,n...,1,1))[:,:,:,1,1]
        gamma_coarse = R(reshape(gamma, n..., 1, 1))[:,:,:,1,1]
        r_coarse = R(real(r)) + im*R(imag(r))

        n_coarse = div.(n,2).+1
        e_coarse = zeros(ComplexF64, n_coarse...,1,1)

        e_coarse = helmholtzVCycle(n_coarse, e_coarse, r_coarse, h.*2, m_coarse, gamma_coarse, omega, R, P;
                            A=A, smoother=smoother, level=level-1, relax_iter=relax_iter, coarse_LiS_solver=coarse_LiS_solver)
        
        fine_error = (P(real(e_coarse)) + im * P(imag(e_coarse)))
        x .+= fine_error

    else
        # coarsest grid
        x_size = size(x)
        if coarse_LiS_solver == nothing
            A_Coarsest(v) =  vec(A(reshape(v,x_size), sl_m, h))
            M_Coarsest(v) = vec(smoother(x, reshape(v,x_size), h, sl_m; A=A, max_iter=1))
            x, flag, err, iter, resvec = fgmres(A_Coarsest, vec(b), 10, tol=1e-15, maxIter=1,
                                                M=M_Coarsest, x=vec(x), out=-1, flexible=true)
        else
            x = LiS_solve(coarse_LiS_solver, b)
        end
        x = reshape(x, x_size)
    end

    x = smoother(x, b, h, sl_m; A=A, max_iter=relax_iter)
    return x
end


# FGMRES iterations
max_iter = 10
restart = 10


n = [64,64,64] .+ 1
m = ones(n...)


h = 1 ./ n
omega = 0.2*pi / (maximum(h)*maximum(sqrt.(m))) # wkh = 0.2pi
# gamma = 0.01 .* ones(n...)
ABLpad = 20
# gamma = getABL(n,true,ones(Int64,3)*ABLpad,Float64(omega)) .+ 0.01#*omega
gamma = ones(n...) .* 0.01*omega
# m[20:50,div(n[1],2)-8:div(n[1],2)+8,div(n[1],2)-8:div(n[1],2)+8] .= 0.01
sl_m, helmholtz_m= getHelmholtzMatrices(m, omega, gamma, h)



b = zeros(ComplexF64, n...)
# b[div(n[1],2),div(n[2],2),1] = 1.0
b[div(n[1],2),div(n[2],2),div(n[3],2)] = 1.0

level = 2
relax_iter = 1

R,P = getInterp(n)

solveLinearSystem(x,b,n) = helmholtzVCycle(n, x, b, h, m, gamma, omega, R, P; A=secondOrderHelmholtz, smoother=helmholtzJacobi, level=level, relax_iter=relax_iter, coarse_LiS_solver=nothing) 
b = reshape(b, size(b)..., 1, 1)
A = v->vec(secondOrderHelmholtz(reshape(v, size(b)), helmholtz_m, h))
M = v->vec(solveLinearSystem(zeros(ComplexF64, size(b)), reshape(v, size(b)), n))
u,flag,err,iter,resvec = @time fgmres(A, vec(b), restart, tol=1e-6, maxIter=max_iter, M=M, x=vec(zeros(ComplexF64, size(b))), out=-1, flexible=true)

u = reshape(u,n...)
println("iterations = $(length(resvec)) with error=$(err)\n")

# plotSlices(m)
plotSlices(u)


