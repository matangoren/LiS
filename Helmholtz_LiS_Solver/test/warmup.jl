


n = [64,64].*1 .+ 1;
# n = [257,1025]
m_coeff = [0.25, 1.0]
lower = m_coeff[1]
upper = m_coeff[end]
m = linear_grid_ratio(lower, upper, n; data_type=data_type)
domain_m = [0.0,1,0.0,1]



# imshow(m); colorbar();
Minv = getRegularMesh(domain_m,collect(size(m)) .- 1);

h = Minv.h

println("########## h = $(h) ##########")
omega = real(getMaximalFrequency(m,Minv));
pad = 20*ones(Int64,length(n));
ABLamp = omega;
println("omega is ",omega/pi," times pi")
println("grid size $(size(m))")

gamma_0 = 0.01
gamma = getABL(Minv.n.+1,false,pad,Float64(omega)) .+ gamma_0*omega


b = zeros(ComplexF64, n...)
b[div(n[1],4)+1,div(n[2],4)+1] = 1.0


T = "High"
orderNeumannBC = 1
add_sommerfeld = true


fine_laplacian_type = T
fine_mass_type = T

level = 3


laplacian_types = laplacian_types_2D
mass_types = mass_types_2D
fine_laplacian_stencil = laplacian_types[fine_laplacian_type]
fine_mass_stencil = mass_types[fine_mass_type]

shift = 0.0
if T == "Low"
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

using Distributed


NumCells = [1,1].*2;
overlap = [4,4].*4;

# num_workers = max(div(prod(NumCells),4),1)

# # Now add fresh workers
# addprocs(num_workers)
println("========= Workers $(workers()) =========")
println("========= Workers $(length(workers())) =========")


sl_matrix, helmholtz_matrix, correction = getHelmholtzMatrices(m, omega, gamma, h, fine_laplacian_stencil,fine_mass_stencil, alpha=alphas[end], add_sommerfeld=add_sommerfeld,BC=orderNeumannBC)

AmatVec = v->vec(HelmholtzOperator(reshape(v, size(b)...,1,1), helmholtz_matrix,correction, h, fine_laplacian_stencil, fine_mass_stencil))

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

# H = GetHelmholtzOperator(Hparam)[1
HrT = sparse(H')
# HrT = H

#Shift = GetHelmholtzShiftOP(m, omega,0.1);
#Shift = convert(SparseMatrixCSC{ComplexF64,spIndType},Shift);





# solver = getJuliaSolver()
solver = getConvGeometricMultigridSolver(;laplace_type=T,mass_type=T,level=level, interType=T, dim=length(n),add_sommerfeld=add_sommerfeld, orderNeumannBC=orderNeumannBC)

DDparam = getDomainDecompositionParam(ComplexF64,Int64,Minv,NumCells,overlap,getNodalIndicesOfCell,solver);

println("##### workers: $(workers()) #####")
println("Performing Absorbing+Neumann Setup")
println("NumCells = ",NumCells," overlap = ",overlap);

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
	if Overlap[1] > overlap[1]
		subgamma = zeros(ComplexF64,IIp_shape)
		# t = false
	end
    Hparam = HelmholtzParam(subMesh,Hparam.gamma[IIp] + subgamma[:],Hparam.m[IIp],Hparam.omega,false,t)

	return Hparam;
end


# getDDMass = (ddp,hp,i)->(0.0.*Vector(diag(GetHelmholtzShiftOP(hp.m,0.0,0.0))));

# function getDirichletMassNodalMesh(DDparam::DomainDecompositionParam,problem_param::HelmholtzParam,i::Array{Int64})
# 	d = getDirichletMassNodal(DDparam.numDomains,DDparam.overlap,i,DDparam.Mesh.n);
# 	d.*=(0.1*4.0)/prod(DDparam.Mesh.h);
# 	return d;
# end
# getDDMass = getDirichletMassNodalMesh

# in comment: working example (From Eran's code)
# Ctor = DomainDecompositionOperatorConstructor{ComplexF64,Int64}(Hparam,getSubParams,GetHelmholtzOperator,identity);
# DDparam = setupDDSerial(HrT,DDparam);

# when using ConvGeometricMultigridSolver
Ctor = DomainDecompositionOperatorConstructor{ComplexF64,Int64}(Hparam,getSubParams,identity,identity);
if length(workers()) <= 1
	DDparam = setupDDSerial(Ctor,DDparam);
else
	println("### IN PARALLEL ###")
	DDparam = setupDDParallel(Ctor,DDparam, workers());
end

println("Performing DD Solution with GMRES")
x = copy(q); 
x[:] .= 0.0;


x = solveLinearSystem!(AmatVec,q,x,DDparam)[1];
# x = @time solveLinearSystem!(HrT,q,x,DDparam)[1];
println("Outside error: ", norm(HrT'*x - q)/norm(q))
println("Outside error: ", norm(vec(AmatVec(reshape(x,n...,1,1))) - q)/norm(q))


