function dampedJac(A,b,w,x,tol,maxit)

    D = diag(A); # vector
    wDinv = w ./ D;
    r = b - A * x;
    iter = 0;

    for i = 1:maxit
        if (norm(r) > tol)
            x .+= wDinv .* r;
            r = b - A * x;
            iter = iter + 1;
        end
    end

    return x, iter
end