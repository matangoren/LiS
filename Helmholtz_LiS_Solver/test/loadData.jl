# using DelimitedFiles

# print(@__DIR__)
# A = readdlm("$(@__DIR__)/GeoModels/MarmousiVp_small.dat");

using DelimitedFiles

function getModel(n)



    if length(n)==3
        error("This code supports only 2D Marmousi");
    end

    domain_data = [0.0,16,0.0,4]; ## that's the original domain

    

    Vp = readdlm("$(@__DIR__)/GeoModels/MarmousiVp_small.dat")/1000.0
    
    n_data = collect(size(Vp));
    M = getRegularMesh(domain_data,n_data);
    
    Vp = expandModelNearest(smoothModel(Vp,[],0),n_data,n);

    n_data = collect(size(Vp));
    
    
    pad_down = 16;

    n_new = tuple((collect(size(Vp)) + [0;pad_down])...);
    Vp_new = zeros(n_new);Vp_new[:,1:size(Vp,2)] = Vp;
    for k=0:pad_down-1
        Vp_new[:,end-k] = Vp[:,end];
    end

    Vp = Vp_new;
    domain_data = [0.0,16.0,0.0,4.0 + pad_down*M.h[2]];

    n_data = collect(size(Vp));
    M = getRegularMesh(domain_data,n_data);
	
	
	return M,Vp

end


function getWedge(bottom, top, n);

	if length(n) == 3
		println("This code supports only 2D wedge media")
	elseif ~ (n[1] == n[2])
		println("n must be square")
	end

    nx = n[1]; ny = n[1];
    x = (0:nx - 1) ./ (nx - 1);
    y = (0:ny - 1) ./ (ny - 1);  
    X = x * ones(Float64, nx)';
    Y = ones(Float64,nx) * y';

    Z = 0.25 .* (tanh.((4 .* Y - X .- 0.75) .* 20)) .+ 0.75; 
    Z[:, end - div(ny, 2) + 1:end] = Z[:, div(ny, 2) : -1 : 1];
    
    ratios = Z .^ 2;

    ratios = ratios .* ((top - bottom) / (1 - 0.25)); # stretch
	top_temp = 1 * ((top - bottom) / (1 - 0.25));
	ratios = ratios .+ (top - top_temp); # translation

    return ratios;
end


function getLinearModel(top,bottom,n)
	if length(n)==2
		rho2 = collect(range(top,stop=bottom,length=n[2]));
		rho1 = ones(n[1]);
		rho = rho1*rho2'
	else
		rho_t = collect(range(top,stop=bottom,length=n[2]));
		rho = ones(tuple(n...));
		for k=1:n[3]
			rho[:,:,k].*=rho_t[k];
		end
	end
	return rho;
end


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

# add if on size(m) for dimension and make the two functions into one

function smoothModel(m,Mesh,times = 0)
	
	if length(size(m)) == 2
		# ms = addAbsorbingLayer2D(m,times);
		ms = copy(m)
		for k=1:times
			for j = 2:size(ms,2)-1
				for i = 2:size(ms,1)-1
					@inbounds ms[i,j] = (2*ms[i,j] + (ms[i-1,j-1]+ms[i-1,j]+ms[i-1,j+1]+ms[i,j-1]+ms[i,j+1]+ms[i+1,j-1]+ms[i+1,j]+ms[i,j+1]))/10.0;
				end
			end
		end
		ms = ms[(times+1):(end-times),1:end-times];
	elseif length(size(m)) == 3
		ms = copy(m)
		mt = copy(m)
		n = size(m);
		for t=1:times
			for k = 1:n[3]
				km1 = max(k-1,1);
				kp1 = min(k+1,n[3]);
				for j = 1:n[2]
					jm1 = max(j-1,1);
					jp1 = min(j+1,n[2]);
					for i = 1:n[1]
						im1 = max(i-1,1);
						ip1 = min(i+1,n[1]);
						@inbounds mt[i,j,k] = (2*ms[i,j,k] + (ms[im1,j,k] + ms[i,jm1,k] + ms[i,j,km1] + ms[ip1,j,k] + ms[i,jp1,k] + ms[i,j,kp1]))/8.0;
					end
				end
			end
			ms[:] = mt[:];
		end
	end
		
	return ms;
end
