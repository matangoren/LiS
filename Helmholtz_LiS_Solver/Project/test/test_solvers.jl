using Helmholtz
using PyPlot
using Printf
using KrylovMethods
using LinearAlgebra

include("../solver.jl")
include("../auxiliary.jl")
include("../LiS/utils.jl")
include("../utils.jl")

function get_coarse_m(m, level, R)
    m_coarse = copy(m)
    for i=1:(level-1)
        m_coarse = R(reshape(m_coarse,size(m_coarse)...,1,1))[:,:,1,1]
    end
    
    return m_coarse
end


initial_n = [64, 64]
# m_coeff = [[1.0]]#, [0.25,1.0]]#, [0.5, 1.0], [0.75,1.0]]

m_coeff = [[1.0]]
# FGMRES iterations
max_iter = 15
restart = 10

close()

for i=3:3
    n = 2^(i-1) .* initial_n .+ 1
    h = 1 ./ n
    println("===== testing for $(n[1])x$(n[2]) grid =====")

    for c in m_coeff
        lower = upper = c[1]
        if length(c) == 2
            upper = c[2]
        end

        println("$(length(c)==1 ? "const $(lower)" : "linear($(lower),$(upper))") test")
        m = linear_grid_ratio(lower, upper, n[1])
        
        println(size(m))
        # figure()
        # imshow(reshape(real(m), n...)); colorbar();

        omega = 0.2*pi / (maximum(h)*maximum(sqrt.(m))) # wkh = 0.2pi
        # gamma = getABL(n,false,ones(Int64,2)*20,Float64(1)) .+ 0.01
        gamma = ones(size(m)) .* 0.01*omega

        b = zeros(ComplexF64, n...)
        b[div(n[1],2),div(n[2],2)] = 1.0

        _, helmholtz_m= getHelmholtzMatrices(m, omega, gamma, h)
        



        for gamma_0 in [0.01] #[0.01, 0.05, 0.1, 0.3, 0.5]
            level = 2
            relax_iter = 1
            R,P = getInterp(length(n); highOrder=false)

            # build coarse level LiS solver (only once!)
            m_coarse = get_coarse_m(m,level, R) 
            h_coarse = h .* (2^(level-1))
            n_coarse = collect(size(m_coarse))
            gamma_0 = (maximum(m) - minimum(m)) < 0.1 ? 0.01 : (maximum(m) - minimum(m))./maximum(m)

            println("gamma_0 $(gamma_0)")
            gamma_coarse = ones(n_coarse...) .* gamma_0*(omega)
            m0_s = (omega)^2 .* [mean(m_coarse)] .* (1 - im*(gamma_0 + 0.5))
            delta_coarse = zeros(ComplexF64, n_coarse...)
            delta_coarse[div(n_coarse[1],2),div(n_coarse[2],2)] = 1.0

            sl_m_coarse, _ = getHelmholtzMatrices(m_coarse, omega, gamma_coarse, h_coarse)
            # coarse_LiS_solver = getLiS_solver(n_coarse, h_coarse, delta_coarse, m0_s, sl_m_coarse, n_coarse)

            
            laplacian_stencils = [[0 -1 0; -1 4 -1; 0 -1 0],[0 -1 0; -1 4 -1; 0 -1 0]] # can be later on changed to ["LOW","LOW"]
            mass_stencils = [[0 0 0; 0 1 0; 0 0 0],[0 0 0; 0 1 0; 0 0 0]]
            alphas = [0.1,0.1]
            solver_MG = getMG_solver(R, P, laplacian_stencils, mass_stencils, alphas, helmholtzJacobi, h, m, gamma, omega, level, relax_iter; coarse_LiS_solver=nothing)

            x, iterations, error = solve(solver_MG, n, b, h, m, gamma, omega, restart, max_iter);
            println("V($(level)) \t iterations = $(iterations) with error=$(error)\n")

            # solver_MG.coarse_LiS_solver = coarse_LiS_solver
            # x, iterations, error = solve(solver_MG, n, b, h, m, gamma, omega, restart, max_iter);
            # println("V($(level))+Lis \t iterations = $(iterations) with error=$(error)\n")

            figure()
            imshow(reshape(real(x), n...)); colorbar();

            # m0_s = omega^2 .* [mean(m)] .* (1 - im*(gamma_0))
            # solver_LiS = getLiS_solver(n, h, b, m0_s, helmholtz_m, n)
            # x, iterations, error = solve(solver_LiS, n, b, h, m, gamma, omega, restart, max_iter);
            # println("LiS \t iterations = $(iterations) with error=$(error)\n")


            
        end

        #=
        # [min,max] test
        m0_s = omega^2 .* [minimum(m),maximum(m)] .* (1 + im*gamma_0)
        solver_LiS = getLiS_solver(n, h, δ, m0_s, helmholtz_m, n)
        x, iterations, error = solve(solver_LiS, n, b, h, m, gamma, omega, restart, max_iter);
        println("[min,max] LiS \t iterations = $(iterations) with error=$(error)\n")

        # [mean] test
        m0_s = omega^2 .* [mean(m)] .* (1 + im*gamma_0)
        solver_LiS = getLiS_solver(n, h, δ, m0_s, helmholtz_m, n)
        x, iterations, error = solve(solver_LiS, n, b, h, m, gamma, omega, restart, max_iter);
        println("[mean] LiS \t iterations = $(iterations) with error=$(error)\n")

        # [m(1/4), m(3/4)] test
        m1 = 0.75*minimum(m) + 0.25*maximum(m)
        m2 = 0.25*minimum(m) + 0.75*maximum(m)
        m0_s = omega^2 .* [m1,m2] .* (1 + im*gamma_0)
        solver_LiS = getLiS_solver(n, h, δ, m0_s, helmholtz_m, n)
        x, iterations, error = solve(solver_LiS, n, b, h, m, gamma, omega, restart, max_iter);
        println("[m(1/4), m(3/4)] LiS \t iterations = $(iterations) with error=$(error)\n")

        # [m(1/3), m(2/3)] test
        m1 = (2/3)*minimum(m) + (1/3)*maximum(m)
        m2 = (1/3)*minimum(m) + (2/3)*maximum(m)
        m0_s = omega^2 .* [m1,m2] .* (1 + im*gamma_0)
        solver_LiS = getLiS_solver(n, h, δ, m0_s, helmholtz_m, n)
        x, iterations, error = solve(solver_LiS, n, b, h, m, gamma, omega, restart, max_iter);
        println("[m(1/3), m(2/3)] LiS \t iterations = $(iterations) with error=$(error)\n")
        =#

        
        
    end
    
end