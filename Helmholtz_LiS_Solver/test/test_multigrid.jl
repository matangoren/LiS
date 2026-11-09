include("./test_setup.jl")

restart = 10
max_iter = 20



# for level in [3]
    level = 3
    relax_iter = [1,1]
    cycle = 2
    coarse_maxIter = 1
    coarse_restart = 10

    shift = 0.0
    if fine_laplacian_type == "Low"
        if level == 3
            shift = 0.3
        elseif level == 4
            shift = 0.5
        end
    elseif fine_laplacian_type == "High"
        if level == 3
            shift =  0.1
        elseif level == 4
            shift = 0.3
        end
    end

    Rs_Grid, Rs, Ps, laplacian_stencils, mass_stencils = buildCycle(level, [T,T,T], fine_laplacian_type, fine_mass_type; dim=length(n))
    
    alphas =  ones(Float64,level) * shift

    println("****************************************************")
    println("grid size: $(n)")
    println("levels: $(level)")
    println("Laplacian order: $(fine_laplacian_type), Mass order: $(fine_mass_type)")
    println("Intergrid order: $(T)")
    println("shift: $(shift)")
    println("relax iterations: $(relax_iter)")
    println("****************************************************")

    _, helmholtz_matrix = getHelmholtzMatrices(m, omega, gamma, h, alpha=alphas[end], add_sommerfeld=add_sommerfeld,BC=orderNeumannBC) #alphas[end]

    solver_MG = getMG_solver(Rs_Grid, Rs, Ps, laplacian_stencils, mass_stencils, alphas, h, m, gamma, omega, level, relax_iter; coarse_LiS_solver=nothing, gmres_maxIter=coarse_maxIter, gmres_restart=coarse_restart, cycle=cycle,orderNeumannBC=orderNeumannBC,add_sommerfeld=add_sommerfeld)

    # println("Run 1: W$(level) with GMRES: $(solver_MG.gmres_maxIter) maxIter and $(solver_MG.gmres_restart) restart")
    # x, iterations, error1 = @time solve(solver_MG, n, b, h, helmholtz_matrix, fine_laplacian_stencil, fine_mass_stencil, restart, max_iter);
    # println("\t iterations = $(iterations) with error=$(error1)\n")


    # solver_MG.gmres_maxIter = 20
    # solver_MG.gmres_restart = 10
    # println("Run 2: W$(level) with GMRES: $(solver_MG.gmres_maxIter) maxIter and $(solver_MG.gmres_restart) restart")
    # x, iterations, error1 = @time solve(solver_MG, n, b, h, helmholtz_matrix, fine_laplacian_stencil, fine_mass_stencil, restart, max_iter);
    # println("\t iterations = $(iterations) with error=$(error1)\n")



    println("Run 3: W$(level) with LiS")
    m_coarse = get_coarse_m(m,level, Rs)
    h_coarse = h .* (2^(level-1))
    n_coarse = collect(size(m_coarse))

    shift = (maximum(m) - minimum(m)) < 0.1 ? alphas[1] : max((1-mean(m))/(maximum(m) - minimum(m))^2, alphas[1])
    println(shift, gamma_0)
    m0_s = (omega)^2 .* [mean(m_coarse)] .* (1 - im*(gamma_0+shift))
    delta_coarse = zeros(ComplexF64, n_coarse...)
    if length(n_coarse) == 2
        delta_coarse[div(n_coarse[1],2)+2,div(n_coarse[2],2)+2] = 1.0
    else
        delta_coarse[div(n_coarse[1],2)+2,div(n_coarse[2],2)+2,div(n_coarse[3],2)+2] = 1.0
    end
    coarse_LiS_solver = getLiS_solver(squeezeConvResult(laplacian_stencils[1]),squeezeConvResult(mass_stencils[1]), n_coarse, h_coarse, delta_coarse, m0_s, div.(n_coarse,2), div.(n_coarse,2))
    solver_MG.coarse_LiS_solver = coarse_LiS_solver
    
    # Minv = getRegularMesh([0.0,1,0.0,1],collect(size(m)) .- 1);
    # pad = 20*ones(Int64,Minv.dim);
    # ABLamp = omega
    # H = GetHelmholtzOperator(Minv,m,omega,gamma,false,pad,ABLamp,true)[1];
    # HrT = sparse(H')
    # x, iterations, error1 = @time solve(HrT, solver_MG, n, b, h, helmholtz_matrix, fine_laplacian_stencil, fine_mass_stencil, restart, max_iter);
    
    println("n/2")
    x, iterations, error1 = @time solve(solver_MG, n, b, h, helmholtz_matrix, fine_laplacian_stencil, fine_mass_stencil, restart, max_iter);
    println("\t iterations = $(iterations) with error=$(error1)\n")
    # # figure()
    # # imshow(reshape(real(x), n...)); colorbar();

    # println("4")
    # coarse_LiS_solver = getLiS_solver(squeezeConvResult(laplacian_stencils[1]),squeezeConvResult(mass_stencils[1]), n_coarse, h_coarse, delta_coarse, m0_s, [4,4,4], [4,4,4])
    # solver_MG.coarse_LiS_solver = coarse_LiS_solver    
    # x, iterations, error1 = @time solve(solver_MG, n, b, h, helmholtz_matrix, fine_laplacian_stencil, fine_mass_stencil, restart, max_iter);
    # println("\t iterations = $(iterations) with error=$(error1)\n")
# end