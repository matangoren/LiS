using jInv.Mesh
using jInv.LinearSolvers
using Test
using KrylovMethods
using Multigrid
using LinearAlgebra
using SparseArrays

println("===  Example 2D DivSigGrad ====");

domain = [0.0, 1.0, 0.0, 1.0];
n      = [50,50];
Mr     = getRegularMesh(domain,n)
G      = getNodalGradientMatrix(Mr);
Ar     = G'*G;
Ar     = Ar + 1e-2*norm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));
N      = size(Ar,2); 
B      = Ar*rand(N,4);

levels      = 5;
numCores 	= 8; 
maxIter     = 15;
relativeTol = 1e-2;
relaxType   = "SPAI";
relaxParam  = 1.0;
relaxPre 	= 2;
relaxPost   = 2;
cycleType   ='V';
coarseSolveType = "Julia";

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType);

sMGPCG   = getMGsolver(MG,Mr, 1,"GMRES",out=1);
X,  = solveLinearSystem(Ar,B,sMGPCG);
@test norm(Ar*X-B)/norm(B) < sMGPCG.tol

sMGPCG   = getMGsolver(MG,Mr, 1,"PCG",out=1);
X,  = solveLinearSystem(Ar,B,sMGPCG);
@test norm(Ar*X-B)/norm(B) < sMGPCG.tol