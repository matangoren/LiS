include("dampedJac.jl")

function MGcycle(A,b,x,w,nu1,nu2,levels,recursive_calls,R_arr,P_arr,Ac_arr,LUAcoarsest; coarseSolve = "exact", n, relaxType = "Jacobi", inner_coarse, maxit_coarse, tol_coarse)

	# for Vcycle, recurcive_calls = 1
    # for Wcycle, recursive_calls = 2
    # for two-grid, levels = 2

	if levels == 1
        if coarseSolve == "exact"
		    return LUAcoarsest \ b
        elseif coarseSolve == "gmres"
            println("coarse solve iterations")
            M_Coarsest(v) = vec(dampedJac(A,v,0.8,(0.0 + 0.0*1im)*zeros(size(v)),1e-6,1)[1])
            e = fgmres(A, b, inner_coarse; maxIter = maxit_coarse, out = -1, tol = tol_coarse , flexible = true)[1]
            println("finished coarse solve iterations")
            return e
        elseif coarseSolve == "jac"
            tol = 1e-6;
            e,_ = dampedJac(A,b,0.2,(0.0 + 0.0*1im)*zeros(size(b)),tol,10);
            return e
        elseif coarseSolve == "GS"
            tol = 1e-6;
            e,_ = GaussSeidel(A,b,(0.0 + 0.0*1im)*zeros(size(b)),tol,5);
            return e
        end
	end
	
	# pre-smoothing
    tol = 1e-5;
    if relaxType == "Jacobi"
        x,_ = dampedJac(A,b,w[1],x,tol,nu1);
    elseif relaxType == "Vanka" # cell-wise, 4 nodes patch
        x = Vanka(A,b,w[1],x,n,tol,nu1);
    elseif relaxType == "GS"
        x,_ = GaussSeidel(A,b,x,tol,nu1);
    end


    # compute and restrict the residual
    r = b - A * x;

    R = R_arr[1];
    P = P_arr[1];
    Ac = Ac_arr[1]

    R_arr = R_arr[2:end];
    P_arr = P_arr[2:end];
    Ac_arr = Ac_arr[2:end];

    n = div.(n,2);

    rc = R * r;

    # solve the error-residual equation directly or recursively
    ec = 0.0 .* rc; # initial guess
	if levels == 2
		recursive_calls = 1;
	end
    for j=1:recursive_calls
        ec = MGcycle(Ac,rc,ec,w[2:end],nu1,nu2,levels-1,recursive_calls,R_arr,P_arr,Ac_arr,LUAcoarsest ; coarseSolve, n, relaxType, inner_coarse, maxit_coarse, tol_coarse);
    end

    e = P * ec;
    x = x + e;

    if relaxType == "Jacobi"
        x,_ = dampedJac(A,b,w[1],x,tol,nu2);
    elseif relaxType == "Vanka"
        x = Vanka(A,b,w[1],x,n,tol,nu2);
    elseif relaxType == "GS"
        x,_ = GaussSeidel(A,b,x,tol,nu2);
    end

    return x

end
