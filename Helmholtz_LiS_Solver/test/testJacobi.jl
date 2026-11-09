using Distributed
using ArgParse
using MAT
using JLD2








@everywhere begin
	coarse_times = [];
	include("./test_import.jl")
end





# # Now add fresh workers

println("========= Workers $(workers()) =========")
println("========= Workers $(length(workers())) =========")

@everywhere begin
	coarse_times = [];
	include("./test_setup.jl")
end

@everywhere function getTest(n;model="linear")

	println(model)

	if model == "marmousi"
		A = readdlm("$(@__DIR__)/GeoModels/MarmousiVp_small.dat", data_type);
		m = expandModelNearest(A*1e-3, size(A),n);
		m = 1 ./ (m.^2)
		m = Matrix(transpose(m))
		n = collect(size(m))
		domain_m = [0.0,1,0.0,div(size(m,2)-1,size(m,1)-1)].*4
	elseif model == "tunnel"
		n = [96,96,48] .+ 1;
		A = read(joinpath(@__DIR__, "TunnelModels", "tunnel_network.bin"))
		A = data_type.(reshape(A,n...))
		vals = reshape(range(2000, 2200; length=n[3]), 1, 1, :);
		m = convert.(data_type, vals .+ zeros(data_type, n[1], n[2], n[3]));
		m[Bool.(A)] .= 340; # to verify.
		m .*= 1e-3;
		m = 1 ./ (m.^2);
		domain_m = [0.0,1,0.0,1,0.0,0.2] # just for now
		
	else # "linear"
		lb = 0.25
		if model == "const"
			lb = 1.0
		end
		m_coeff = [lb, 1.0]
		lower = m_coeff[1]
		upper = m_coeff[end]
		if length(n) == 2
			m = linear_grid_ratio(lower, upper, n; data_type=data_type)
			domain_m = [0.0,1,0.0,1]
		else
			# 3D linear model
			vals = range(lower, upper; length=n[2])
			m = convert.(data_type, reshape(vals, 1, n[2], 1) .* ones(data_type, n[1], 1, n[3]))
			domain_m = [0.0,1,0.0,1,0.0,1]
		end
	end

	Minv = getRegularMesh(domain_m,collect(size(m)) .- 1);

	h = Minv.h

	println("########## h = $(h) ##########")
	omega = real(getMaximalFrequency(m,Minv))*4;
	# omega = 0.001
	# omega = real(getMaximalFrequency(m,Minv)*0.9);
	pad = 10*ones(Int64,length(n));
	ABLamp = omega;
	println("omega is ",omega/pi," times pi")
	println("grid size $(size(m))")

	gamma = getABL(Minv.n.+1,false,pad,Float64(ABLamp)) .+ 0.1*omega

	b = zeros(ComplexF64, n...)
	if length(n) == 2
		b[div(n[1],2)+1,div(n[2],2)+1] = 1.0
	else
		b[div(n[1],2)+1,div(n[2],2)+1, div(n[3],2)+1] = 1.0
	end

	return m,Minv,h,omega,gamma,b
end



function pmr(A!::Function, M!::Function, b::AbstractVector{T};
             x0=nothing, tol=1e-8, maxiter=1000, verbose=false) where {T}
    x = x0 === nothing ? zeros(T, length(b)) : copy(x0)
    r = b - A!(x);          # r = b - A x
    z = similar(b); q = similar(b)
    nb = norm(b); nb == 0 && (nb = one(real(T)))
    hist = [norm(r) / nb]
    for k in 1:maxiter
        hist[end] < tol && break
        z = M!(r) #M!(z, r)                                     # z = M r
        q = A!(z) #A!(q, z)                                     # q = A M r
        qq = real(dot(q, q))
        qq == 0 && break
        α = dot(q, r) / qq                           # minimizes ‖r - α q‖
		println("alpha $(α)")
        axpy!(α, z, x)
        axpy!(-α, q, r)
        push!(hist, norm(r) / nb)
        verbose && println("iter $k: relres = $(hist[end])")
    end
    return x, hist
end


    i = 32





	n = [i,i] .+ 1;
	

	m,Minv,h,omega,gamma,b = getTest(n;model="const")


    # alpha = 0
    # gamma = zeros(size(gamma))

	sl_matrix, helmholtz_matrix, correction = getHelmholtzMatrices(m, omega, gamma, h, fine_laplacian_stencil,fine_mass_stencil, alpha=alpha, add_sommerfeld=false,BC=1)
	


    AmatVec = v->vec(HelmholtzOperator(reshape(v, size(b)...,1,1), helmholtz_matrix,correction, h, fine_laplacian_stencil, fine_mass_stencil))

	sol = rand(n...)
	q = AmatVec(sol)

    x = zeros(Float64,n...,1,1)
    
	x_size = size(x)
	w = 0.8
 	A_Coarsest(v) =  vec(HelmholtzOperator(reshape(v,x_size), helmholtz_matrix, correction, h, fine_laplacian_stencil, fine_mass_stencil))
	M_Coarsest(v) = vec(applyJacobi(reshape(v,x_size), h, helmholtz_matrix, correction, fine_laplacian_stencil, fine_mass_stencil; max_iter=1, w=w))

	maxIter = 2
	restart = 10
	tol = 0.0001
	# x,flag,err,iter,resvec = fgmres_func(A_Coarsest, vec(q), restart, tol=tol, maxIter=maxIter,M=M_Coarsest, x=vec(zeros(n...)), out=-1, flexible=true)
    
	x, hist = pmr(A_Coarsest, M_Coarsest, vec(q);x0=nothing, tol=tol, maxiter=maxIter, verbose=false)

	println("WithJ Outside error: ", norm(vec(AmatVec(reshape(x,n...,1,1))) - q)/norm(q))
	close("all")
	# imshow(real(reshape((vec(AmatVec(reshape(x,n...,1,1))) - q)/norm(q),n...))); colorbar()
	imshow(abs.(real(reshape(vec(x)-vec(sol),n...))),vmin=0.0, vmax=1.0); colorbar()
	savefig("$(@__DIR__)/Error_withJ.png")

	
	# x,flag,err,iter,resvec = fgmres_func(A_Coarsest, vec(q), restart, tol=tol, maxIter=maxIter, x=vec(zeros(n...)), out=-1, flexible=true)
	
	x, hist = pmr(A_Coarsest, identity, vec(q);x0=nothing, tol=tol, maxiter=maxIter, verbose=false)
            
	println("noJ Outside error: ", norm(vec(AmatVec(reshape(x,n...,1,1))) - q)/norm(q))
	close("all")
	imshow(abs.(real(reshape(vec(x)-vec(sol),n...))), vmin=0, vmax=1.0); colorbar()
	savefig("$(@__DIR__)/Error_noJ.png")


	# D = getJacobiDiagonal(h, helmholtz_matrix, correction, fine_laplacian_stencil, fine_mass_stencil);
	# println("Imag")
	# display(imag(D))
	# println("Real ")
	# display(real(D))




# for j=1:10
    #     x = helmholtzJacobi(x, b, h, -helmholtz_matrix, correction, fine_laplacian_stencil, fine_mass_stencil; w=0.8, max_iter=1)
	#     println("Iter #$(j)")
    #     println("Outside error: ", norm(vec(AmatVec(reshape(x,n...,1,1))) - q)/norm(q))
        
    # end
