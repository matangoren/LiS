function intergrid1D(n,isnodal; intergridType = "BI") # Neumann, put the number of cells as n, isnodal is a scalar
    
    if isnodal == true

        # if intergridType == "BI" || intergridType == "mixed" 
        #     P = spdiagm(-1 => 0.5 .* ones(n), 0 => ones(n+1), 1 => 0.5 .* ones(n));
        #     P = P[:,1:2:end];
        # elseif intergridType == "high" 
        #     P = spdiagm(-2 => 0.125 .* ones(n-1), -1 => 0.5 .* ones(n), 0 => 0.75 .* ones(n+1), 1 => 0.5 .* ones(n), 2 => 0.125 .* ones(n-1));
        #     P = P[:,1:2:end];
        # end

        # # keep same ratios in boundaries to get better convergence
        # # P[:,1] = 2 .* P[:,1] ./ sum(P[:,1]);
        # # P[:,end] = 2 .* P[:,end] ./ sum(P[:,end]);
        # # or do nothing to fit the choices we did in the monolithic

        # R = 0.5 .* P';

        P_BI = spdiagm(-1 => 0.5 .* ones(n), 0 => ones(n+1), 1 => 0.5 .* ones(n));
        P_HO = spdiagm(-2 => 0.125 .* ones(n-1), -1 => 0.5 .* ones(n), 0 => 0.75 .* ones(n+1), 1 => 0.5 .* ones(n), 2 => 0.125 .* ones(n-1));

        if intergridType == "mixed_high"
            P = P_HO[:,1:2:end];
            R = 0.5 .* P_BI[:,1:2:end]';
        elseif intergridType == "BI"
            P = P_BI[:,1:2:end];
            R = 0.5 .* P';
        elseif intergridType == "high"
            P = P_HO[:,1:2:end];
            R = 0.5 .* P';
        end

    else

        P = spdiagm(-1 => 0.25 .* ones(n-1), 0 => 0.75 .* ones(n), 1 => 0.75 .* ones(n-1), 2 => 0.25 .* ones(n-2));
        P = P[:,2:2:end];

        # keep same ratios in boundaries to get better convergence
        # P[:,1] = 2 .* P[:,1] ./ sum(P[:,1]);
        # P[:,end] = 2 .* P[:,end] ./ sum(P[:,end]);

        # injection in boundaries to fit the choice we did in the monolithic
        P[1,1] = 1;
        P[end,end] = 1;

        if intergridType == "mixed"
            R = spdiagm(0 => 0.5 .* ones(n), 1 => 0.5 .* ones(n-1));
            R = R[:,2:2:end];
            R = 1.0 .* R';
        else
            R = 0.5 .* P';
        end

    end
    
    return R,P
end


function intergrid(n,isnodal ; intergridType = "BI") # Neumann, put an array of number of cells as n, isnodal is an array

    dim = length(n);

    R,P = intergrid1D(n[1],isnodal[1] ; intergridType);
    R = [R];
    P = [P];
    for i=2:dim
        temp,_ = intergrid1D(n[i],isnodal[i] ; intergridType);
        R = [[temp] ; R];
        _,temp = intergrid1D(n[i],isnodal[i] ; intergridType);
        P = [[temp] ; P];
    end
    if dim > 1
        R = kron(R...);
        P = kron(P...);
    end
    
    return R,P
end
