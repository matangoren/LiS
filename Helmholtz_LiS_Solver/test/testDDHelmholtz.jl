using Distributed
using ArgParse
using MAT
using JLD2


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--DD_n"
            help = "NumCells"
            arg_type = Int
            default = 1

        "--model"
            help = "model type"
            arg_type = String
            default = "tunnel"

		"--coarse_solver"
            help = "coarse solver type"
            arg_type = String
            default = "GMRES_1_10"

		"--tol"
            help = "Multigrid corse gmres tolerance"
            arg_type = Float64
            default = 0.1

		"--dim"
			help = "grid dimension"
			arg_type = Int
			default = 2
    end

    return parse_args(ARGS, s)
end

args = parse_commandline()

DD_n   = args["DD_n"]
model = args["model"]
coarse_solver = args["coarse_solver"]
coarse_tol = args["tol"]
dim = args["dim"]


coarse_gmres_maxIter=20
coarse_gmres_restart=20

coarse_solver_parts = split(coarse_solver,"_")
coarse_solver_type = coarse_solver_parts[1]
coarse_gmres_maxIter = parse(Int, coarse_solver_parts[2])
coarse_gmres_restart = parse(Int, coarse_solver_parts[3])




# mat_filename_jdl = "$(@__DIR__)/dump/Marmousi/1024_4096_DD_4_16.jld2"
mat_filename_jdl = "$(@__DIR__)/dump/Marmousi/512_2048_DD_4_16.jld2"

# const mat_counter = Ref(0)
# const f = jldopen(mat_filename_jdl, "w")

# function save_submatrix(m, solver_matrices)

#     mat_counter[] += 1
#     name = "B_$(mat_counter[])"
# 	if mat_counter[] <= 64
# 		f[name] = (m, solver_matrices)
# 	end

# end


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
		# n = [256,256,96] .+ 1;
		Tunnel = read(joinpath(@__DIR__, "TunnelModels/Example", "tunnel_network_$(n[1])_$(n[2])_$(n[3]).bin"))
		m = ones(Float64, n...)
		m[Bool.(Tunnel)] .= 0.1; # to verify.
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
	omega = real(getMaximalFrequency(m,Minv)); # x1.25 for 8 grid points per wavelength
	# omega = real(getMaximalFrequency(m,Minv)*0.9);
	pad = 20*ones(Int64,length(n));
	ABLamp = omega;
	println("omega is ",omega/pi," times pi")
	println("grid size $(size(m))")

	gamma = getABL(Minv.n.+1,false,pad,Float64(omega)) .+ gamma_0*omega

	b = zeros(ComplexF64, n...)
	if length(n) == 2
		b[div(n[1],2)+1,div(n[2],2)+1] = 1.0
	else
		b[div(n[1],2)+1,div(n[2],2)+1, div(n[3],2)+1] = 1.0
	end

	return m,Minv,h,omega,gamma,b
end

@everywhere function getSubParams(Hparam, M::RegularMesh,i::Array{Int64},NumCells::Array{Int64},Overlap::Array{Int64})
		subMesh   = getSubMeshOfCell(NumCells,Overlap,i,M);
		IIp, IIp_shape = getNodalIndicesOfCell(NumCells,Overlap,i,M.n);
		if length(M.n) == 2
			code 	  = [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2];];
		else
			code 	  = [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2]; i[3]!=1 i[3]!=NumCells[3];];
		end

		subgamma  = getABL(subMesh.n.+1,i[end]==1,Overlap.+4,2.0./(M.h[1]),code) #.+0.001*Hparam.omega;
		t = add_sommerfeld
		if Overlap[1] > 16 #overlap[1]
			subgamma = zeros(ComplexF64,IIp_shape)
			# t = false
		end
		Hparam = HelmholtzParam(subMesh,Hparam.gamma[IIp] + subgamma[:],Hparam.m[IIp],Hparam.omega,false,t)

		return Hparam;
	end







# 2D
if dim == 2
	NumCells_orig = [1,1];
	overlap_orig = [0,0];
	if model == "marmousi"
		NumCells_orig = [max(1,div(DD_n,4)),DD_n];
	else
		NumCells_orig = [DD_n,DD_n];
	end

	if DD_n > 1
		overlap_orig = [4,4].*4;
	end
end


# 3D
if dim == 3
	overlap_orig = [0,0,0];
	NumCells_orig = [DD_n,DD_n, DD_n];
	if DD_n > 1
		overlap_orig = [4,4,4].*4;
	end
end


sizes = []




if model == "linear"
	if dim == 2
		sizes = [256, 1024]
	else
		sizes = [32,256]
	end
end

if model == "marmousi"
	sizes = [1024, 4096]
end

if model == "tunnel"
	sizes = [0]
end


coarse_times = []
coarse_iter = []
coarse_err = []


for (i,s) in enumerate(sizes)
	global coarse_times
	global coarse_iter
	global coarse_err
	
	if i == 1
		if dim == 2
			overlap = [0,0];
			NumCells = [1,1];
		else
			overlap = [0,0,0];
			NumCells = [1,1,1];
		end
	else
		overlap = overlap_orig
		NumCells = NumCells_orig
	end
	n = s
	if model == "marmousi"
		n = [n,div(n,4)] .+ 1;
	else # linear
		if dim == 2
			n = [n,n] .+ 1;
		else
			n = [n,n,n] .+ 1;
		end
	end

	if model == "tunnel"
		if i == 1
			n = [513,513,193]
			# n = [129,129,49]
		elseif i == 2
			n = [257,257,97]
		else
			n = [513,513,193]
		end
	end


	if device == gpu
		result_filename = "gpu_$(join(n,'_'))_$(join(NumCells,'_'))_$(model)_$(coarse_solver)_SPAI_ExactKmeans4_tol-$(coarse_tol)"
	else
		result_filename = "cpu_$(join(n,'_'))_$(join(NumCells,'_'))_$(model)_$(coarse_solver)_tol-$(coarse_tol)"
	end


	coarse_times = []
	coarse_iter = []
	coarse_err = []
	println("############## $(model) ##############")

	m,Minv,h,omega,gamma,b = getTest(n;model=model)

	println("$(coarse_solver) --- $(model) --- $(size(m)) --- $(NumCells)")


	Hparam = HelmholtzParam(Minv,gamma,m,omega,false,add_sommerfeld)
	q = vec(b)



	if T == "Low"
		H, Lap, MM = GetHelmholtzOperator(Hparam,orderNeumannBC);
	else
		beta = 2/3
		if length(n) == 3
			beta = [1/3;1/2]
		end
		H, Lap, MM,Ms = GetHelmholtzOperatorHO(Hparam,beta,orderNeumannBC);
	end

	HrT = sparse(H')

	sl_matrix, helmholtz_matrix, correction = getHelmholtzMatrices(m, omega, gamma, h, fine_laplacian_stencil,fine_mass_stencil, alpha=alphas[end], add_sommerfeld=add_sommerfeld,BC=orderNeumannBC)
	
	AmatVec = v->vec(HelmholtzOperator(reshape(v, size(b)...,1,1), helmholtz_matrix,correction, h, fine_laplacian_stencil, fine_mass_stencil))

	solver = getConvGeometricMultigridSolver(;laplace_type=T,mass_type=T,level=level,gmres_maxIter=coarse_gmres_maxIter,gmres_restart=coarse_gmres_restart, interType=T, dim=length(n),add_sommerfeld=add_sommerfeld, orderNeumannBC=orderNeumannBC,coarse_solver_type=coarse_solver_type, coarse_tol=coarse_tol)
	DDparam = getDomainDecompositionParam(ComplexF64,Int64,Minv,NumCells,overlap,getNodalIndicesOfCell,solver);

	println("##### workers: $(workers()) #####")
	println("Performing Absorbing+Neumann Setup")
	println("NumCells = ",NumCells," overlap = ",overlap);


	# when using ConvGeometricMultigridSolver
	Ctor = DomainDecompositionOperatorConstructor{ComplexF64,Int64}(Hparam,getSubParams,identity,identity);
	if length(workers()) <= 1
		DDparam = setupDDSerial(Ctor,DDparam);
	else
		println("### IN PARALLEL ###")
		DDparam = setupDDParallel(Ctor,DDparam, workers());
	end

	println("Performing DD Solution with $(coarse_solver_type)")
	x = copy(q); 
	x[:] .= 0.0;

	# println("SETUP completed")
	# close(f);
	# return;

	x = solveLinearSystem!(AmatVec,q,x,DDparam)[1];
	# x = @time solveLinearSystem!(HrT,q,x,DDparam)[1];
	println("$(coarse_solver_type) --- $(model) --- $(size(m)) --- $(NumCells)")
	println("Outside error: ", norm(HrT'*x - q)/norm(q))
	println("Outside error: ", norm(vec(AmatVec(reshape(x,n...,1,1))) - q)/norm(q))
	# println(length(coarse_iter))
	
	if device == gpu
		writedlm("$(@__DIR__)/Results/$(model)/GPU/tol_study/$(result_filename).csv", hcat(coarse_times,coarse_iter,coarse_err),',')
	else
		output_dir = "$(@__DIR__)/Results/forPaper/LiSvsCompactLiS/CPU/$(model)"
		mkpath(output_dir)
		writedlm("$(output_dir)/$(result_filename).csv", hcat(coarse_times,coarse_iter,coarse_err),',')

		writedlm("$(output_dir)/x_sol_$(join(n,'_'))_$(coarse_solver).dat", vec(real(x)))
	end	
end