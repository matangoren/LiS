using Helmholtz
using PyPlot
using Printf
using KrylovMethods
using LinearAlgebra

include("../solver.jl")
include("../auxiliary.jl")
include("../LiS/utils.jl")
include("../utils.jl")


initial_n = [64, 64]
m_coeff = [[1.0], [0.6,1.0], [0.4,0.7],[0.25,1.0], [0.5, 1.0], [0.25,0.5]]


# FGMRES iterations
max_iter = 20
restart = 20


for i=3:3
    n = 2^(i-1) .* initial_n .+ 1
    h = 1 ./ n
    println("===== testing for $(n[1])x$(n[2]) grid =====")
    for c in m_coeff[1:1]
        lower = upper = c[1]
        if length(c) == 2
            upper = c[2]
        end

        println("$(length(c)==1 ? "const $(lower)*ones" : "linear($(lower),$(upper))") test")

        m = linear_grid_ratio(lower, upper, n[1])
        close("all")

        # figure()
        # imshow(reshape(real(m), n...)); colorbar();

        omega = 0.2*pi / (maximum(h)*maximum(sqrt.(m))) # wkh = 0.2pi
        gamma = 0.01
        sl_m, helmholtz_m= getHelmholtzMatrices(m, omega, gamma, h)

        δ = zeros(ComplexF64, n...)
        δ[div(n[1],2),div(n[2],2)] = 1.0

        # point-sorce rhs
        b = zeros(ComplexF64, n..., 1,2)
        b[div(n[1],2),div(n[2],2),1,1] = 1.0
        b[div(n[1],2)-20,div(n[2],2)+20,1,2] = 1.0


        gamma_0 = 1.0
        # gamma_0 = (maximum(m) - minimum(m)) < 0.1 ? gamma : (maximum(m) - minimum(m))./maximum(m)
        println(gamma_0)
        m0_s = omega^2 .* [mean(m)] .* (1 + im*gamma_0)

        solver_LiS = getLiS_solver(n, h, δ, m0_s, helmholtz_m, n)
        solver_MG = getMG_solver(secondOrderHelmholtz, helmholtzJacobi, h, m, gamma, omega, 3, 1)
        solver_Multi = getMulti_solver([solver_MG, solver_LiS])


        x, iterations, error = solve(solver_Multi, n, b, h, m, gamma, omega, restart, max_iter);
        println("solver_Multi \t iterations = $(iterations) with error=$(error)\n")

        # figure()
        # imshow(reshape(real(x[:,1]), n...)); colorbar();
        # figure()
        # imshow(reshape(real(x[:,2]), n...)); colorbar();

        
        
    end
    
end