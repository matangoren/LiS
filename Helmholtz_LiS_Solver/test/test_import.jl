using Helmholtz
using PyPlot
using Printf
using KrylovMethods
using LinearAlgebra
using Flux
using DelimitedFiles
using jInv.Mesh;
using Multigrid
using Multigrid.DomainDecomposition
using Multigrid.ParallelJuliaSolver
using SparseArrays
using jInv.LinearSolvers

using CUDA

gpu(x) = cu(x)
cpu(x) = Array(x)

device = CUDA.functional() ? gpu : cpu

println("Using ", CUDA.functional() ? "GPU" : "CPU")

include("../src/gpu_krylov.jl")

cgpu = device

if device == cpu
	fgmres_func = KrylovMethods.fgmres
else
	CUDA.allowscalar(true)
	fgmres_func = gpu_flexible_gmres
end

data_type = Float64
r_type = data_type
gamma_0 = 0.01


include("../src/utils.jl")
include("../src/auxiliary.jl")
include("../src/Modules/Lippmann_solver.jl")
include("../src/operators.jl")


function expandModelNearest(m,n,ntarget)
	if length(size(m))==2
		mnew = zeros(Float64,ntarget[1],ntarget[2]);
		for j=1:ntarget[2]
			for i=1:ntarget[1]
				jorig = convert(Int64,ceil((j/ntarget[2])*n[2]));
				iorig = convert(Int64,ceil((i/ntarget[1])*n[1]));
				mnew[i,j] = m[iorig,jorig];
			end
		end
	elseif length(size(m))==3
		mnew = zeros(Float64,ntarget[1],ntarget[2],ntarget[3]);
		for k=1:ntarget[3]
			for j=1:ntarget[2]
				for i=1:ntarget[1]
					korig = max(1,convert(Int64,floor((k/ntarget[3])*n[3])));
					jorig = convert(Int64,floor((j/ntarget[2])*n[2]));
					iorig = convert(Int64,floor((i/ntarget[1])*n[1]));
					mnew[i,j,k] = m[iorig,jorig,korig];
				end
			end
		end
	end
	return mnew
end