include("intergrid.jl")

function myMGsetup(A,n,levels,isnodal;intergridType = "BI",coarseSolve = "exact")

    n_arr = (0.5 .^ (0:levels-1))' * 1.0 .* n;
    n_arr = Int.(n_arr);

    R1,P1 = intergrid(n,isnodal ; intergridType);
    # println("size R1 ",size(R1),", size A ",size(A),", size(P1)",size(P1))
    Ac1 = R1 * A * P1;

    R_arr = [R1];
    P_arr = [P1];
    Ac_arr = [Ac1];
    for i=2:levels-1
        R_temp,P_temp = intergrid(n_arr[:,i],isnodal ; intergridType);
        R_arr = [R_arr ; [R_temp]];
        P_arr = [P_arr ; [P_temp]];
        
        Ac_temp = R_temp * Ac_arr[i-1] * P_temp;
        Ac_arr = [Ac_arr ; [Ac_temp]];
    end
    
    if coarseSolve == "exact"
        LUAcoarsest = lu(Ac_arr[end]);
    else 
        LUAcoarsest = 0.0;
    end

    return R_arr,P_arr,Ac_arr,LUAcoarsest

end

function MGsetupVectorEq2D(Op,n,levels)

    isnodal_u = [true;false];
    isnodal_v = [false;true];
    isnodal_p = [false;false];
    

    n_arr = (0.5 .^ (0:levels-1))' * 1.0 .* n;
    n_arr = Int.(n_arr);

    R1_u,P1_u = mixedintergrid(n,isnodal_u);
    R1_v,P1_v = mixedintergrid(n,isnodal_v);
    R1_p,P1_p = mixedintergrid(n,isnodal_p);

    R1 = blockdiag(R1_u,R1_v,R1_p);
    P1 = blockdiag(P1_u,P1_v,P1_p);

    Ac1 = R1 * Op * P1;

    R_arr = [R1];
    P_arr = [P1];
    Ac_arr = [Ac1];
    for i=2:levels-1

        R_temp_u,P_temp_u = mixedintergrid(n_arr[:,i],isnodal_u);
        R_temp_v,P_temp_v = mixedintergrid(n_arr[:,i],isnodal_v);
        R_temp_p,P_temp_p = mixedintergrid(n_arr[:,i],isnodal_p);

        R_temp = blockdiag(R_temp_u,R_temp_v,R_temp_p);
        P_temp = blockdiag(P_temp_u,P_temp_v,P_temp_p);

        R_arr = [R_arr ; [R_temp]];
        P_arr = [P_arr ; [P_temp]];
        
        Ac_temp = R_temp * Ac_arr[i-1] * P_temp;
        Ac_arr = [Ac_arr ; [Ac_temp]];
    end
    LUAcoarsest = lu(Ac_arr[end]);

    return R_arr,P_arr,Ac_arr,LUAcoarsest

end




function MGsetupVectorEq3D(Op,n,levels)

    isnodal_u = [true;false;false];
    isnodal_v = [false;true;false];
    isnodal_w = [false;false;true];
    isnodal_p = [false;false;false];
    

    n_arr = (0.5 .^ (0:levels-1))' * 1.0 .* n;
    n_arr = Int.(n_arr);

    R1_u,P1_u = mixedintergrid(n,isnodal_u);
    R1_v,P1_v = mixedintergrid(n,isnodal_v);
    R1_w,P1_w = mixedintergrid(n,isnodal_w);
    R1_p,P1_p = mixedintergrid(n,isnodal_p);

    R1 = blockdiag(R1_u,R1_v,R1_w,R1_p);
    P1 = blockdiag(P1_u,P1_v,P1_w,P1_p);

    Ac1 = R1 * Op * P1;

    R_arr = [R1];
    P_arr = [P1];
    Ac_arr = [Ac1];
    for i=2:levels-1

        R_temp_u,P_temp_u = mixedintergrid(n_arr[:,i],isnodal_u);
        R_temp_v,P_temp_v = mixedintergrid(n_arr[:,i],isnodal_v);
        R_temp_w,P_temp_w = mixedintergrid(n_arr[:,i],isnodal_w);
        R_temp_p,P_temp_p = mixedintergrid(n_arr[:,i],isnodal_p);

        R_temp = blockdiag(R_temp_u,R_temp_v,R_temp_w,R_temp_p);
        P_temp = blockdiag(P_temp_u,P_temp_v,P_temp_w,P_temp_p);

        R_arr = [R_arr ; [R_temp]];
        P_arr = [P_arr ; [P_temp]];
        
        Ac_temp = R_temp * Ac_arr[i-1] * P_temp;
        Ac_arr = [Ac_arr ; [Ac_temp]];
    end
    LUAcoarsest = lu(Ac_arr[end]);

    return R_arr,P_arr,Ac_arr,LUAcoarsest

end



function MGsetupVectorEq2Doriginal(Op,n,levels)

    isnodal_u = [true;false];
    isnodal_v = [false;true];    

    n_arr = (0.5 .^ (0:levels-1))' * 1.0 .* n;
    n_arr = Int.(n_arr);

    R1_u,P1_u = intergrid(n,isnodal_u);
    R1_v,P1_v = intergrid(n,isnodal_v);

    R1 = blockdiag(R1_u,R1_v);
    P1 = blockdiag(P1_u,P1_v);

    Ac1 = R1 * Op * P1;

    R_arr = [R1];
    P_arr = [P1];
    Ac_arr = [Ac1];
    for i=2:levels-1

        R_temp_u,P_temp_u = intergrid(n_arr[:,i],isnodal_u);
        R_temp_v,P_temp_v = intergrid(n_arr[:,i],isnodal_v);

        R_temp = blockdiag(R_temp_u,R_temp_v);
        P_temp = blockdiag(P_temp_u,P_temp_v);

        R_arr = [R_arr ; [R_temp]];
        P_arr = [P_arr ; [P_temp]];
        
        Ac_temp = R_temp * Ac_arr[i-1] * P_temp;
        Ac_arr = [Ac_arr ; [Ac_temp]];
    end
    LUAcoarsest = lu(Ac_arr[end]);

    return R_arr,P_arr,Ac_arr,LUAcoarsest

end