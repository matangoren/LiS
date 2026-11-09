using KrylovMethods
using Helmholtz
using jInv.Mesh
using PyPlot
using SparseArrays
using LinearAlgebra 
close("all")

include("MGsetup.jl")
include("MGcycle.jl")
include("getModels.jl")

n = [128;128]*2; 
Omega = [0.0;1.0;0.0;1.0];
# n = [128;64]; 
# Omega = [0.0;2.0;0.0;1.0];
# n0 = [544;112];
# Omega = [0.0,17.0,0.0,3.5];
# n = [400;128]; 
# n = [800;256];
# n = [1600;512];
# Omega = [0.0;16.0;0.0;5.0];

M = getRegularMesh(Omega,n);

nodes = n + [1;1];
# m = 1.0 * ones(nodes...);
# m = getWedge(0.25,1,nodes);
m = getLinearModel(0.25,1.0,nodes);

omega_factor = 1;
omega = omega_factor * getMaximalFrequency(m,M);
println("omega is ",omega/pi," times pi")


pad = 20;
aten = 0.0;
shift = 0.1;
println("shift is ",shift)
alpha = aten + shift*omega;
neumanOnTop = false;
gamma = getABL(nodes, neumanOnTop, [pad; pad], omega) .+ aten;
gamma_s = getABL(nodes, neumanOnTop, [pad; pad], omega) .+ alpha;
param = HelmholtzParam(M,gamma,m,omega,neumanOnTop,false);
param_s = HelmholtzParam(M,gamma_s,m,omega,neumanOnTop,false);
beta = 2/3;
println("beta is ",beta)
H = GetHelmholtzOperatorHO(param,beta);
H_s = GetHelmholtzOperatorHO(param_s,beta); # 2/3 for 4th order discretization, 1 for 2nd order

# q,src = getAcousticPointSource(M,Float64);
# b = vec(q);

b = zeros(ComplexF64, nodes...)
b[div(n[1],2)+1,div(n[2],2)+1] = 1.0
b = vec(b)


#### direct solve and plot to validate
# sol = H\b;
# sol = reshape(sol,size(m));
# imshow(real(sol)')


#### multigrid preconditioner

relaxType = "Jacobi"
relaxParam = [0.8; 0.8; 0.8] # [0.89;0.89;0.7];


nu1 = 1;
nu2 = 1;
nodal = [true;true]; # matan, don't touch, you only need true true
levels = 3;
recursive_calls = 2; # 1 for V-cycle, 2 for W-cycle
# R_arr,P_arr,Ac_arr,LUcoarsest = myMGsetup(H_s,n,levels,nodal; intergridType = "BI"); # for 121 intergrid
R_arr,P_arr,Ac_arr,LUcoarsest = myMGsetup(H_s,n,levels,nodal; intergridType = "high"); # for 14641 intergrid



function PrecFuncSL(r) # shifted Laplacian MG prec to use in GMRES

    e = MGcycle(H_s,r,0.0*r,relaxParam,nu1,nu2,levels,recursive_calls,R_arr,P_arr,Ac_arr,LUcoarsest; n, coarseSolve = "gmres", inner_coarse, maxit_coarse, tol_coarse);

    return e
end



##### GMRES solver with MG as a preconditioner #####

inner = 5; # restart for outer GMRES
maxIter = 20; # number of outer iterations in outer GMRES
tol = 1e-6; # tolerance for outer GMRES
inner_coarse = 10; # restart for GMRES coarse solver
maxit_coarse = 20; # outer iterations for GMRES coarse solver
tol_coarse = 1e-2; # tolerance for GMRES coarse solver

e = fgmres(H, (1.0 + 0.0*im)*b, inner; maxIter, M = PrecFuncSL, out = 2, tol = tol , flexible = true)[1];

sol = reshape(e,size(m));

figure()
imshow(real(sol)')