

# Assuming test_import is including this file - device is defined there.

using LinearAlgebra

"""
    getSPAIprec(m, laplacian_stencil, mass_stencil, h)

Compute the diagonal SPAI preconditioner for

    A(x) = laplacian_conv(pad_repeat(x, rL)) +
           mass_conv(pad_repeat(m .* x, rM))

Returns Q such that Q[i] = conj(A[i,i]) / sum_j(abs2(A[i,j])).

Assumptions:
- Odd-sized, centered stencils.
- Repeated-edge padding.
- Matching stencil orientation between this routine and Conv.
- The only operator terms are the two stencil convolutions shown above.
"""
function getSPAIprec(m, laplacian_stencil, mass_stencil, h)
    N = ndims(m)
    size(laplacian_stencil) |> length # validate stencil object

    @assert ndims(laplacian_stencil) == N
    @assert ndims(mass_stencil) == N
    @assert all(iseven.(size(laplacian_stencil)) .== false)
    @assert all(iseven.(size(mass_stencil)) .== false)

    T = promote_type(
        eltype(m),
        eltype(laplacian_stencil),
        eltype(mass_stencil),
        Float64
    )

    L = laplacian_stencil ./ h[1]^2
    M = mass_stencil

    rL = CartesianIndex(ntuple(d -> div(size(L, d), 2), N))
    rM = CartesianIndex(ntuple(d -> div(size(M, d), 2), N))

    # Dictionary entries are the effective coefficients of a row,
    # indexed by the corresponding input-grid point.
    Q = similar(m, T)

    for I in CartesianIndices(m)
        row = Dict{CartesianIndex{N}, T}()

        # Laplacian contributions.
        for K in CartesianIndices(L)
            offset = K - rL
            J = CartesianIndex(ntuple(
                d -> clamp(I[d] + offset[d], 1, size(m, d)), N
            ))

            row[J] = get(row, J, zero(T)) + L[K]
        end

        # Mass contributions: M * (m .* x).
        for K in CartesianIndices(M)
            offset = K - rM
            J = CartesianIndex(ntuple(
                d -> clamp(I[d] + offset[d], 1, size(m, d)), N
            ))

            row[J] = get(row, J, zero(T)) + M[K] * m[J]
        end

        # Exact squared norm of the effective row.
        s = sum(abs2, values(row))

        # Diagonal coefficient.
        aii = get(row, I, zero(T))

        Q[I] = iszero(s) ? zero(T) : conj(aii) / s
    end

    return Q
end


function JacobiDiagonalCorrection2D(shape, stencil; code=nothing)
    if code == nothing
        code = ones(Bool, 2*length(shape))
    end
    s = div(size(stencil,1),2)
    correction = zeros(shape)
    mid = stencil[s+1,s+1]
 
    if code[1] # top row
        correction[1,:] .+= sum(stencil[1:s,s+1])
        if code[3] # top-left corner
            correction[1,1] += sum(stencil[1:s,1:s])
        end
        if code[4] # top-right corner
            correction[1,end] += sum(stencil[1:s,end-s+1:end])
        end
    end
    if code[2] # bottom row
        correction[end,:] .+= sum(stencil[end-s+1:end,s+1])
        if code[3] # bottom-left corner
            correction[end,1] += sum(stencil[end-s+1:end,1:s])
        end
        if code[4] # bottom-right corner
            correction[end,end] += sum(stencil[end-s+1:end,end-s+1:end])
        end
    end
    if code[3] # left column
        correction[:,1] .+= sum(stencil[s+1,1:s])
    end
    if code[4] # right column
        correction[:,end] .+= sum(stencil[s+1,end-s+1:end])
    end

    return correction
end


function JacobiDiagonalCorrection3D(shape, stencil; code=nothing)
    if code == nothing
        code = ones(Bool, 2*length(shape))
    end
    s = div(size(stencil,1),2)
    correction = zeros(shape)
    mid = stencil[s+1,s+1,s+1]

    if code[1] # top layer
        correction[1,:,:] .+= sum(stencil[1:s,s+1,s+1])
        if code[3] && code[5]
            correction[1,1,1] += sum(stencil[1:s,1:s,1:s])
        end
        if code[3] && code[6]
            correction[1,1,end] += sum(stencil[1:s,1:s,end-s+1:end])
        end
        if code[4] && code[5]
            correction[1,end,1] += sum(stencil[1:s,end-s+1:end,1:s])
        end
        if code[4] && code[6]
            correction[1,end,end] += sum(stencil[1:s,end-s+1:end,end-s+1:end])
        end
    end
    if code[2] # bottom layer
        correction[end,:,:] .+= sum(stencil[end-s+1:end,s+1,s+1])
        if code[3] && code[5]
            correction[end,1,1] += sum(stencil[end-s+1:end,1:s,1:s])
        end
        if code[3] && code[6]
            correction[end,1,end] += sum(stencil[1:s,1:s,end-s+1:end])
        end
        if code[4] && code[5]
            correction[end,end,1] += sum(stencil[end-s+1:end,end-s+1:end,1:s])
        end
        if code[4] && code[6]
            correction[end,end,end] += sum(stencil[end-s+1:end,end-s+1:end,end-s+1:end])
        end
    end
    if code[3] # left layer
        correction[:,1,:] .+= sum(stencil[s+1,1:s,s+1])
    end
    if code[4] # right layer
        correction[:,end,:] .+= sum(stencil[s+1,end-s+1:end,s+1])
    end
    if code[5] # front layer
        correction[:,:,1] .+= sum(stencil[s+1,s+1,1:s])
    end
    if code[6] # back layer
        correction[:,:,end] .+= sum(stencil[s+1,s+1,end-s+1:end])
    end

    return correction
end

# function getJacobiDiagonal(h, matrix, correction, laplacian_stencil, mass_stencil)
#     lap_mid = div(size(laplacian_stencil,1),2)+1
#     mass_mid = div(size(mass_stencil,1),2)+1
#     if length(h) == 2
#         D = laplacian_stencil[lap_mid, lap_mid]/(h[1]^2) .+ mass_stencil[mass_mid,mass_mid].*matrix #(-matrix .+ correction[1])
#     else
#         D = laplacian_stencil[lap_mid, lap_mid,lap_mid]/(h[1]^2) .+ mass_stencil[mass_mid,mass_mid,lap_mid].*matrix #(-matrix .+ correction[1])
#     end

#     D .+= correction[1] .+ correction[2].*matrix
    
#     return D
# end

function getHelmholtzMatrices(m, omega, gamma, h, laplacian_stencil, mass_stencil; alpha=0.1, add_sommerfeld=true, BC=1,code=nothing)
    sommerfeld = zeros(ComplexF64, size(m))
    if code == nothing
        code = ones(Bool, 2*length(h))
    end
    if add_sommerfeld
        if ndims(sommerfeld) == 2
            if code[1]
                sommerfeld[1,:]  .+= ((BC/h[1]) .* im*omega*sqrt.(m[1,:]))
            end
            if code[2]
                sommerfeld[end,:]  .+=  ((BC/h[1]) .* im*omega*sqrt.(m[end,:]))
            end
            if code[3]
                sommerfeld[:,1]  .+=   ((BC/h[2]) .* im*omega*sqrt.(m[:,1]))
            end
            if code[4]
                sommerfeld[:,end] .+=  ((BC/h[2]) .* im*omega*sqrt.(m[:,end]))
            end
        else
            if code[1]
                sommerfeld[1,:,:]  .+= ((BC/h[1]) .* im*omega*sqrt.(m[1,:,:]))
            end
            if code[2]
                sommerfeld[end,:,:]  .+= ((BC/h[1]) .* im*omega*sqrt.(m[end,:,:]))
            end
            if code[3]
                sommerfeld[:,1,:]  .+= ((BC/h[2]) .* im*omega*sqrt.(m[:,1,:]))
            end
            if code[4]
                sommerfeld[:,end,:] .+= ((BC/h[2]) .* im*omega*sqrt.(m[:,end,:]))
            end
            if code[5]
                sommerfeld[:,:,1]  .+=  ((BC/h[3]) .* im*omega*sqrt.(m[:,:,1]))
            end
            if code[6]
                sommerfeld[:,:,end] .+= ((BC/h[3]) .* im*omega*sqrt.(m[:,:,end]))
            end
        end
    end

  
    if length(h) == 2
        laplacian_correction = JacobiDiagonalCorrection2D(size(m), laplacian_stencil ./ (h[1]^2); code=code)
        mass_correction = JacobiDiagonalCorrection2D(size(m),mass_stencil; code=code)
    else
        laplacian_correction = JacobiDiagonalCorrection3D(size(m), laplacian_stencil ./ (h[1]^2); code=code)
        mass_correction = JacobiDiagonalCorrection3D(size(m),mass_stencil; code=code)
    end

    helmholtz_matrix = -(omega.^2).*(m).*(1.0.-1im.*gamma/omega) .+ sommerfeld
    sl_matrix = -(omega.^2).*(m).*(1.0.-1im.*(gamma/omega .+ alpha)) .+ sommerfeld

    correction = [laplacian_correction,mass_correction]

    Q = getSPAIprec(sl_matrix, squeezeConvResult(laplacian_stencil), squeezeConvResult(mass_stencil), h)


    # sl_matrix_Jacobi_D = getJacobiDiagonal(h, sl_matrix, correction, laplacian_stencil, mass_stencil)

    return sl_matrix, helmholtz_matrix, correction, Q
end

function squeezeConvResult(x)
    n_dims = ndims(x)

    # Specify the last two dimensions to drop
    dims_to_drop = (n_dims - 1, n_dims)

    # Drop the dimensions
    return dropdims(x, dims=dims_to_drop)
end

function get_coarse_m(m, level, Rs_Grid)
    m_coarse = reshape(copy(m), size(m)...,1,1)
    for i=(level-1):-1:1
        m_coarse = Rs_Grid[i](m_coarse)
    end
    
    return squeezeConvResult(m_coarse)
end


function getMatrices(Rs_grid, m, omega, gamma, h,laplacian_stencils, mass_stencils; level=3, alphas=0.1, add_sommerfeld=true, BC=1, code=nothing)
    if !isa(alphas,AbstractVector)
        alphas = ones(data_type,level).*0.1
    end
    matrices = Vector{}(undef, level)
    m = copy(m)
    gamma = copy(gamma)
    h = copy(h)
    matrices[level] = getHelmholtzMatrices(m, omega, gamma, h, laplacian_stencils[level], mass_stencils[level]; alpha=alphas[level], add_sommerfeld=add_sommerfeld, BC=BC, code=code)
    for i=(level-1):-1:1
        R_grid = Rs_grid[i]
        m = squeezeConvResult(R_grid(reshape(m,size(m)...,1,1)))
        gamma = squeezeConvResult(R_grid(reshape(gamma,size(gamma)...,1,1)))
        h.*=2
        matrices[i] = getHelmholtzMatrices(m, omega, gamma, h, laplacian_stencils[i], mass_stencils[i]; alpha=alphas[i], add_sommerfeld=add_sommerfeld, BC=BC, code=code)
    end
    return matrices
end


using Statistics

"""
    kmeans1d_exact(x, k)

Globally optimal 1D K-means clustering using dynamic programming.

Returns:
    centers      : cluster means, ordered increasingly
    assignments  : cluster index for each original input point
    objective    : total within-cluster sum of squared errors

Complexity:
    Time:   O(k*n^2)
    Memory: O(k*n)

This is the straightforward exact DP algorithm, not the paper's
faster O(k*n) monotone-matrix-search implementation.
"""
function kmeans1d_exact(x::AbstractVector{<:Real}, k::Integer)

    n = length(x)

    @assert n > 0 "Input vector must not be empty."
    @assert 1 <= k <= n "Require 1 <= k <= length(x)."
    @assert all(isfinite, x) "Input values must be finite."

    # Sort values while retaining their original indices.
    perm = sortperm(x)
    xs = Float64.(x[perm])

    # Shift values to reduce cancellation in prefix-sum calculations.
    xs .-= xs[1]

    # Prefix sums: index 1 represents zero elements.
    prefix_sum = zeros(Float64, n + 1)
    prefix_sq  = zeros(Float64, n + 1)

    for j in 1:n
        prefix_sum[j + 1] = prefix_sum[j] + xs[j]
        prefix_sq[j + 1] = prefix_sq[j] + xs[j]^2
    end

    # SSE of the sorted interval xs[i:j], in O(1).
    function cluster_cost(i, j)
        s = prefix_sum[j + 1] - prefix_sum[i]
        q = prefix_sq[j + 1] - prefix_sq[i]
        return max(0.0, q - s^2 / (j - i + 1))
    end

    # D[c+1, j+1] = minimum SSE for j points in c clusters.
    D = fill(Inf, k + 1, n + 1)
    split = zeros(Int, k + 1, n + 1)

    D[1, 1] = 0.0

    for c in 1:k
        for j in c:n
            best_cost = Inf
            best_i = 0

            for i in c:j
                candidate = D[c, i] + cluster_cost(i, j)

                if candidate < best_cost
                    best_cost = candidate
                    best_i = i
                end
            end

            D[c + 1, j + 1] = best_cost
            split[c + 1, j + 1] = best_i
        end
    end

    # Recover the optimal clusters in sorted order.
    sorted_assignments = zeros(Int, n)
    centers = zeros(Float64, k)
    j = n

    for c in k:-1:1
        i = split[c + 1, j + 1]

        sorted_assignments[i:j] .= c
        centers[c] = mean(xs[i:j]) + Float64(x[perm[1]])
        j = i - 1
    end

    # Restore assignments to the original input order.
    assignments = similar(sorted_assignments)

    for p in 1:n
        assignments[perm[p]] = sorted_assignments[p]
    end

    return (
        centers = centers,
        assignments = assignments,
        objective = D[k + 1, n + 1]
    )
end


