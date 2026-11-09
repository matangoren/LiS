using Random
using Distributions
using StatsBase
using Printf
using Distributions

function getUniformM0s(lower, upper; num=4)
    # get #num values of m0 in the range (lower, upper)
    m0s = rand(Uniform(lower,upper), num)
end

# Grid ratio initialization

function wedge_grid_ratio(bottom, top, n); # no regular mesh
    nx = n; ny = n;
    x = (0:nx - 1) ./ (nx - 1);
    y = (0:ny - 1) ./ (ny - 1);  
    X = x * ones(Float64, nx)';
    Y = ones(Float64,nx) * y';

    Z = 0.25 .* (tanh.((4 .* Y - X .- 0.75) .* 20)) .+ 0.75; 
    Z[:, end - div(ny, 2) + 1:end] = Z[:, div(ny, 2) : -1 : 1];
    
    ratios = Z .^ 2;

    # Original wedge model: bottom=0.25, top=1.
    # We want to strech and move the range of values
    # to get a wedge model with any wanted range
    ratios = ratios .* ((top - bottom) / (1 - 0.25)); # stretch
	top_temp = 1 * ((top - bottom) / (1 - 0.25));
	ratios = ratios .+ (top - top_temp); # translation

    return ratios';
end

function linear_grid_ratio(bottom, top, n; data_type=Float64)
    # x = (0:n - 1) ./ (n - 1);
    # x = x .* (top - bottom) .+ bottom; 
    # ratios = x * ones(Float64, n)'; 

    rho2 = collect(data_type,range(bottom,stop=top,length=n[1]));
	rho1 = ones(data_type, n[2]);
	rho = rho1*rho2'
	return permutedims(rho);
end

"""
Return ratio-grid with two different intensities, such that the intensity p1 surrounds a n/2*n/2 sized square with 
intensity p2.
"""
function dual_grid_ratio(bottom, top, n)
    ratios = ones(ComplexF64, n, n) .* bottom
    ratios[Int(n/4)+1: Int(3n/4), Int(n/4)+1:Int(3n/4)] = ones(Int(n/2), Int(n/2)) .* top
    return ratios;
end

function split_grid_ratio(bottom, top, n)
    ratios = ones(ComplexF64, n, n) .* top
    ratios[1: div(n,2), :] .= bottom
    return ratios;
end

function interpolated_split_grid_ratio(bottom, top, n)
    scale = div(n,1)+1              # scale need to be odd.
    sc2 = div(scale , 2)            # sc2 need to be even.
    ratios = ones(ComplexF64, n+scale-1, n+scale-1) .* top
    ratios[1:div(n+scale, 2), :] .= bottom
    kernel = ones(scale) / scale 
    for i in sc2+1:n+sc2
        for j in 1:n+scale-1
            ratios[i,j] = sum(ratios[i-sc2:i+sc2,j] .* kernel)
        end
    end
    return ratios[sc2+1:n+sc2,sc2+1:n+sc2];
end

"""Generate a ratio-grid with 3 ratios: p1, p2 and p3."""
function triple_grid_ratio(bottom, middle, top, n)
    ratios = ones(ComplexF64, n, n) .* bottom
    ratios[div(2n,12)+1: div(10n,12), div(2n,12)+1:div(10n,12)] .= middle 
    ratios[div(5n,12)+1: div(7n,12), div(5n,12)+1:div(7n,12)] .= top
    return ratios;
end

"""Generate a ratio-grid with 3 ratios: p1, p2 and p3 (p1 on the outer layer, then p2 and p3 in the inner layers)"""
function octagon_grid_ratio(p1, p2, p3, p4, p5, p6, p7, p8, n)
    ratios = zeros(ComplexF64, n, n) .+ p1
    ratios[div(2n,16): div(15n,16), div(2n,16): div(15n,16)] .= p2
    ratios[div(3n,16): div(14n,16), div(3n,16):div(14n,16)] .= p3 
    ratios[div(4n,16): div(13n,16), div(4n,16):div(13n,16)] .= p4 
    ratios[div(5n,16): div(12n,16), div(5n,16):div(12n,16)] .= p5 
    ratios[div(6n,16): div(11n,16), div(6n,16):div(11n,16)] .= p6 
    ratios[div(7n,16): div(10n,16), div(7n,16):div(10n,16)] .= p7
    ratios[div(8n,16): div(9n,16), div(8n,16):div(9n,16)] .= p8 
    return ratios;
end

"""Generate a constant ratio-grid of ones"""
function const_grid_ratio(n) 
    return ones(ComplexF64, n, n);
end

"""Generate a randomly sampled ratio-grid"""
function random_grid_ratio(n)
    return rand(n, n)
end

"""
n - grid size (n X n matrix)
p - intencity of delta
return a grid with a delta in the center of the grids.
"""
function delta_grid_ratio(p, n)
    ratios = ones(ComplexF64, n, n)
    ratios[Int(n/2), Int(n/2)] *= p
    return ratios;
end

"""
num - number of deltas
p1 - intencity of delta
p2 - intencitiy of the background
n - grid size (n X n matrix)
returns a grid with num deltas.
"""
function deltas_grid_ratio(num, p1, p2, n)
    if num > n * n
        num = n * n
    end
    indices = sample(1:n * n, num, replace = false)
    ratios = p2 .* ones(ComplexF64, n * n)
    for i in 1:num
        ratios[indices[i]] = p1
    end
    return reshape(ratios, n, n);
end


"""Returns a grid with gaussian distribution."""
function gaussian_grid_ratio(mu, sigma, n)
    d = Normal(mu, sigma)
    ratios = rand(d, n, n)
    return ratios
end

# m(x,y) Initialization
"""Generates a vector with a linear range of components from the minimal value of a ratio-grid to the maximum, 
    calculated for the fgmres function"""
function linear_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    min_m, max_m = get_value(m_grid, findmin), get_value(m_grid, findmax);
    delta = (real(max_m) - real(min_m)) / (max_iter * restrt) + abs(imag(max_m) - imag(min_m))im / (max_iter * restrt);
    m_0_reals = collect((i for i in real(min_m):real(delta):real(max_m)))
    m_0_ims = collect((i for i in imag(min_m):imag(delta):imag(max_m)))
    m_0s = zeros(ComplexF64, size(m_0_reals)[1])
    for i in 1:(size(m_0s)[1]-1)
        m_0s[i] = m_0_reals[i] + m_0_ims[i]im
    end
    return m_0s;
end

# """Returns a vector of length max_iter * restrt, 
#     filled with random numbers in the range [min_val, max_val] of the ratio-grid ratio."""
# function random_min_max_m(m_base, ratio, max_iter, restrt)
#     m_grid = m_base * ratio
#     min_m, max_m = get_value(m_grid, findmin), get_value(m_grid, findmax)
#     return rand(Uniform(real(min_m), real(max_m)), max_iter * restrt) .+ 
#     1im * rand(Uniform(imag(min_m), imag(max_m)), max_iter * restrt);
# end

"""Returns a vector of length max_iter * restrt,
    where elements are randomly sampled from ratio WITHOUT REPITITIONS."""
function random_no_rep_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    return sample(m_grid[:], max_iter * restrt, replace = false)
end

"""Returns a vector of length of max_iter * restrt,
    where elements are randomly sampled from ratio with repetitions."""
function monte_carlo_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    return sample(m_grid[:], max_iter * restrt)
end

"""
returns a vector of length of max_iter * restrt,
filled with the average value of the ratio-grid ratio.
"""
function avg_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    avg_m = sum(m_grid) / (size(m_grid)[1] * size(m_grid)[2])
    return avg_m * ones(ComplexF64, max_iter * restrt);
end

"""
return a vector of length of max_iter * restrt,
with an ordered range.
"""
function gaussian_range_m(m_base, ratio, max_iter, restrt)
    effective_sigma = 0.05
    d = fit(Normal, real.(ratio[:]))
    lo, hi = quantile.(d, [0.5-effective_sigma, 0.5+effective_sigma])
    x = range(lo, hi; length = max_iter * restrt)
    # samples = pdf.(d, x)
    return m_base * x
end

function gaussian_depricated_m(m_base, ratio, max_iter, restrt)
    sigma_ratio = 0.05
    d = fit(Normal, real.(ratio[:]))
    sigma = std(d)
    new_d = Normal(mean(d), sigma*sigma_ratio)
    return m_base .* rand(new_d, max_iter * restrt)
end

"""
return a vector of length of max_iter * restrt,
where elements are sampled from minimum value and maximum value of the 
ratio-grid ratio, in a tethered manner.
"""
function min_max_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    min_m, max_m = get_value(m_grid, findmin), get_value(m_grid, findmax);
    m_0s = zeros(ComplexF64, max_iter * restrt)
    for i in 1:size(m_0s)[1]
        m_0s[i] = i % 2 == 0 ? min_m : max_m
    end
    return m_0s;
end

"""
return a vector of length of max_iter * restrt,
where elements are the average of the average value of the 
ratio-grid ratio, and a randomly sampled value from ratio.
"""
function combined_monte_carlo_avg(m_base, ratio, max_iter, restrt)
    avg = avg_m(m_base, ratio, max_iter, restrt)
    monte_carlo = monte_carlo_m(m_base, ratio, max_iter, restrt)
    return (avg+monte_carlo) / 2
end

# other functions
"""get element from matrix, given an operator (findmin/findmax)."""
function get_value(A, operator)
    _, indices = operator(norm.(A))
    i = indices[1]
    j = indices[2]
    return A[i,j];
end

"""Compute and return the standard-error of the values of vector arr."""
function compute_stderr(arr, len)
    avg = sum(arr) / len
    std_err = arr[:] .- avg * ones(len)
    return sqrt(sum(std_err .* std_err) / len)
end

"""Print results."""
function print_result(res, names_dict, grid_name)
    s = @sprintf "Sequence Results for %s:\n" grid_name 
    for (i, (j, num_of_iteration, val)) in enumerate(res)
        s = @sprintf "%s%d. %-30s ---> Number of iteration: %d | Value: %0.3e\n" s i names_dict[j] num_of_iteration val
    end
    @printf "%s" s
end

"""Print results."""
function print_result_with_err(res, names_dict, grid_name)
    s = @sprintf "Sequence Results for %s:\n" grid_name 
    for (i, (j, num_of_iteration, val, std_err)) in enumerate(res)
        s = @sprintf "%s%d. %-30s ---> Number of iteration: %d | Value: %0.3e | Stderr: %0.3f\n" s i names_dict[j] num_of_iteration val std_err
    end
    @printf "%s" s
end
