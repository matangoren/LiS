using Flux



function getLapacianConv(h::Vector{Float64}; center_coeff=-1, edge_coeff=0, pad=1)
    h1 = center_coeff / (h[1]^2)
    h2 = center_coeff / (h[2]^2)    
    edge_h1 = edge_coeff / (h[1]^2)
    edge_h2 = edge_coeff / (h[2]^2)

    stencil = Float64.(reshape([edge_h1 h1 edge_h1;h2 -2*(h1+h2+edge_h1+edge_h2) h2;edge_h1 h1 edge_h1],3,3,1,1))

    return Conv(stencil, zeros(Float64, 1); pad=pad)
end

# function secondOrderHelmholtz(x::Array{ComplexF64}, matrix, h::Vector{Float64})
#     # stencil = Float64.((1 / (h[1]*h[2])) * [0 -1 0; -1 4 -1; 0 -1 0]);
#     # Laplacian = Conv(reshape(stencil,3,3,1,1), zeros(Float64, 1); pad=1)
#     Laplacian = getLapacianConv(h)
#     return Laplacian(x) - x.*matrix
# end

# function fourthOrderHelmholtz(x::Array{ComplexF64}, matrix::Array{ComplexF64}, h::Vector{Float64})
#     # stencil = Float64.((1 / (h[1]*h[2])) * [-1/6 -2/3 -1/6; -2/3 10/3 -2/3; -1/6 -2/3 -1/6]);
#     # Laplacian = Conv(reshape(stencil,3,3,1,1), zeros(Float64, 1); pad=1)
#     Laplacian = getLapacianConv(h; center_coeff=(-2/3), edge_coeff=(-1/6))

#     mass_stencil = reshape([0 1/12 0; 1/12 2/3 1/12; 0 1/12 0],3,3,1,1)

#     return Laplacian(x) - matrix.*Conv(mass_stencil, zeros(1); pad=1)(x) # verify secoond term
# end

function HelmholtzOperator(x::Array{ComplexF64}, mass::Array{ComplexF64}, h::Vector{Float64}, laplacian_stencil, mass_stencil)
    laplacian_conv = Conv(reshape(laplacian_stencil ./ (h[1])^2, size(laplacian_stencil)...,1,1), zeros(Float64, 1), pad=div(size(laplacian_stencil,1),2));
    mass_conv = Conv(reshape(mass_stencil, size(mass_stencil)...,1,1), zeros(Float64, 1), pad=div(size(mass_stencil,1),2));
    
    return laplacian_conv(x) - mass.*mass_conv(x)
end


