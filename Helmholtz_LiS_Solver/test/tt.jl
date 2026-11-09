using SparseArrays
using Flux
using ArrayPadding

include("./test_setup.jl")
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


x = rand(Float64,n...,1,1);
t1 = AmatVec(x)
t2 = H*vec(x);
fig = figure()
imshow(real(reshape(t1.-t2,n...))); colorbar();
title("Operator")
fig.savefig("Op.png")
println("Op diff norn $(norm(real(reshape(t1.-t2,n...))))")

ff = div(size(fine_mass_stencil,1),2)
mass_conv = Conv(fine_mass_stencil, zeros(Float64, 1), pad=ff);
g = helmholtz_matrix.*x
t1 = vec(mass_conv(g) .+ correction[2].*g)
t2 = MM*vec(x)
fig = figure();
imshow(real(reshape(t1.-t2,n...))); colorbar();
title("Mass")
fig.savefig("Mass.png")
println("Mass diff norn $(norm(real(reshape(t1.-t2,n...))))")


laplacian_conv = Conv(fine_laplacian_stencil ./ (h[1]^2), zeros(Float64, 1), pad=0);
t1 = vec(laplacian_conv(pad_repeat(x,div(size(fine_laplacian_stencil,1),2))))
t2 = Lap*vec(x)
fig = figure();
imshow(real(reshape(t1.-t2,n...))); colorbar();
title("Laplacian")
fig.savefig("Laplacian.png")
println("Laplacian diff norn $(norm(real(reshape(t1.-t2,n...))))")




# figure();
# imshow(real(reshape(t1.-t2,n...)[20:end-20,20:end-20])); colorbar();

export getNodalLaplacianMatrix, multOpNeumann!, Lap2DStencil,dxxMat


function getBC(orderNeumannBC)
BC = 2.0;
if orderNeumannBC == 2
	#one is for first order Neumann, and 2 is for 2nd order. Note that with 2, the matrix is not symmetric!!!
	BC = 2.0;
elseif orderNeumannBC == 1
	BC = 1.0;
else
	println("getNodalLaplacianMatrix: BC not supported")
end
return BC;
end


function dxxMat(n::Int64,h::Float64,orderNeumannBC=2)

BC = getBC(orderNeumannBC);
O1 = -ones(n-1);
O1[n-1] = -BC;
O2 = 2.0*ones(n);
O2[1] = BC;
O2[n] = BC;
O3 = -ones(n-1);
O3[1] = -BC;
dxx = spdiagm(-1 => O1./(h^2), 0 => O2./(h^2), 1=> O3./(h^2))
return dxx;
end

function getNodalLaplacianMatrix(Msh::RegularMesh,orderNeumannBC=2)
nodes = Msh.n .+ 1;
I1 = SparseMatrixCSC(1.0I, nodes[1], nodes[1]);
Dxx1 = dxxMat(nodes[1],Msh.h[1],orderNeumannBC);
I2 = SparseMatrixCSC(1.0I, nodes[2], nodes[2]);
Dxx2 = dxxMat(nodes[2],Msh.h[2],orderNeumannBC);
if Msh.dim==2
	L = kron(I2,Dxx1) + kron(Dxx2,I1);
else
	I3 = SparseMatrixCSC(1.0I, nodes[3], nodes[3]);
	Dxx3 = dxxMat(nodes[3],Msh.h[3],orderNeumannBC);
	L = kron(I3,kron(I2,Dxx1) .+ kron(Dxx2,I1)) .+ kron(Dxx3,kron(I2,I1));
end
return L;
end


function speye(n)
	return sparse(1.0I,n,n);
end

function ddxCN(n,h)
# D = ddx(n), 1D derivative operator
	I,J,V = SparseArrays.spdiagm_internal(0 => fill(-(1/h),n), 1 => fill((1/h),n)) 
	return sparse(I, J, V, n, n+1)
end

function av3term(n::Int64,alpha=5/6)
	t = (1-alpha)/2;
	T = spdiagm(0=>alpha*ones(n), 1=>t*ones(n-1),-1=>t*ones(n-1));
	T[1,1] = 1/2 + alpha/2;
	T[end,end] = 1/2 + alpha/2;
	return T;
end


function getNodalSpreadGradients(Msh,avFunc)
	n = Msh.n;
	h = Msh.h;

    if length(n) == 2

        tmp = ddxCN(n[1],h[1]);
        D1 = kron(speye(n[2]+1),tmp);
        D1s = kron(avFunc(n[2]+1),tmp);

        tmp = ddxCN(n[2],h[2])
        D2 = kron(tmp,speye(n[1]+1))
        D2s = kron(tmp,avFunc(n[1]+1))

        G = [D1;D2];
        Gs = [D1s;D2s];

    elseif length(n) == 3

		tmp = ddxCN(n[1],h[1]);
        D1 = kron(speye(n[3]+1),kron(speye(n[2]+1),tmp));
        D1s = 0.5*(kron(speye(n[3]+1),kron(avFunc(n[2]+1),tmp)) + kron(avFunc(n[3]+1),kron(speye(n[2]+1),tmp)));

        tmp = ddxCN(n[2],h[2]);
        D2 = kron(speye(n[3]+1),kron(tmp,speye(n[1]+1)));
        D2s = 0.5*(kron(speye(n[3]+1),kron(tmp,avFunc(n[1]+1))) + kron(avFunc(n[3]+1),kron(tmp,speye(n[1]+1))));

        tmp = ddxCN(n[3],h[3]);
        D3 = kron(tmp,kron(speye(n[2]+1),speye(n[1]+1)));
        D3s = 0.5*(kron(tmp,kron(speye(n[2]+1),avFunc(n[1]+1))) + kron(tmp,kron(avFunc(n[2]+1),speye(n[1]+1))));

        G = [D1;D2;D3];
        Gs = [D1s;D2s;D3s];
    end

    return G,Gs
end

function getSpreadNodalLaplacianAndMass(Mesh,beta)
	# for 2D beta is a scalar and works both for the Laplacian and mass
	# for 3D beta is a vector, where beta[1] is for the Laplacian and beta[2] for the mass

    # Grad  = getNodalGradientMatrix(Msh) 
    # Lap   = Grad'*Grad
    n = Mesh.n;
    if length(n) == 2

        avFunc = n -> av3term(n,0.5);
        G,Gs = getNodalSpreadGradients(Mesh,avFunc);
        Gs = (1-beta)*Gs+beta*G;
        Lap = G'*Gs;

        M = 0.5*kron(av3term(n[2]+1,beta),speye(n[1]+1)) + 0.5*kron(speye(n[2]+1),av3term(n[1]+1,beta));

    elseif length(n) == 3

		if beta == 1
			beta = [1;1]
		end

        avFunc = n -> av3term(n,0.5);
        G,Gs = getNodalSpreadGradients(Mesh,avFunc);
        Gs = (1-beta[1])*Gs+beta[1]*G;
        Lap = G'*Gs;

        third = (1.0/3.0);
        M = third*(kron(speye(n[3]+1),kron(av3term(n[2]+1,beta[2]),speye(n[1]+1))) + 
                   kron(speye(n[3]+1),kron(speye(n[2]+1),av3term(n[1]+1,beta[2]))) + 
                   kron(av3term(n[3]+1,beta[2]),kron(speye(n[2]+1),speye(n[1]+1))));

    end
    
    return Lap,M
end

export restrictCellCenteredVariables,restrictNodalVariables2,getFWInterp;


# Bilinear "Full Weighting" prolongation
function getFWInterp(n_nodes::Array{Int64,1},geometric::Bool=false)
# n here is the number of NODES!!!
(P1,nc1) = get1DFWInterp(n_nodes[1],geometric);
(P2,nc2) = get1DFWInterp(n_nodes[2],geometric);
if length(n_nodes)==3
	(P3,nc3) = get1DFWInterp(n_nodes[3],geometric);
end
if length(n_nodes)==2
	P = kron(P2,P1);
	nc = [nc1,nc2];
else
	P = kron(P3,kron(P2,P1));
	nc = [nc1,nc2,nc3];
end
return P,nc
end

function get1DFWInterp(n_nodes::Int64,geometric)
# n here is the number of NODES!!!
oddDim = mod(n_nodes,2);
if n_nodes > 2
	halfVec = 0.5*ones(n_nodes-1);
	P = spdiagm(-1=>halfVec,0=>ones(n_nodes),1=>halfVec); #used to be P = spdiagm((halfVec,ones(n_nodes),halfVec),[-1,0,1],n_nodes,n_nodes);
    if oddDim == 1
        P = P[:,1:2:end];
    else
		if geometric
			P = sparse(1.0I,n_nodes,n_nodes);
			println("Warning: getFWInterp(): in geometric mode we stop coarsening because num cells does not divide by two");
		else 
			P = P[:,[1:2:end;end]];
			P[end-1:end,end-1:end] = speye(2);
#         	P = P[:,1:2:end];
#         	P[end,end-1:end] = [-0.5,1.5];
		end
    end
else
    P = sparse(1.0I,n_nodes,n_nodes);
end
nc = size(P,2);
return P,nc
end



########################### GEOMETRIC MULTIGRID STUFF ###############################

function restrictCellCenteredVariables(rho::Array,n::Array{Int64})
R,nc = getRestrictionCellCentered(n);
rho_c = (0.5^length(n))*(R*rho[:]);
R.nzval .*= (0.5^length(n));
return rho_c,R;
## TODO: make this more efficient...
end

export restrictNodalVariables
function restrictNodalVariables(rho::Array,n_nodes::Array{Int64})
P,nc = getFWInterp(n_nodes,true);
rho_c = zeros(eltype(rho),size(P,2));
rho_c[:] = (0.5^length(n_nodes))*(P'*rho[:]);
return rho_c;
## TODO: make this more efficient...
end

function restrictNodalVariables2(rho::Array,n_nodes::Array{Int64})
# P,nc = getFWInterp(n_nodes,true);
R1,nc1 = get1DNodeFullWeightRestriction(n_nodes[1]-1);
R2,nc2 = get1DNodeFullWeightRestriction(n_nodes[2]-1);
R = kron(R2,R1);
R.nzval .*= (0.5^length(n_nodes));
# rho_c = zeros(eltype(rho),size(P,2));
# println(size(R))
# println(size(rho[:]))

rho_c = R*rho[:];
return rho_c,R;
## TODO: make this more efficient...
end



