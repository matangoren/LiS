using Flux


function HelmholtzOperator(x::AbstractArray, mass::AbstractArray, correction, h::Vector{Float64}, laplacian_stencil, mass_stencil)
    lap_bias = similar(laplacian_stencil, 1)
    fill!(lap_bias, 0)
    mass_bias = similar(mass_stencil, 1)
    fill!(mass_bias, 0)

    laplacian_conv = Conv(laplacian_stencil ./ (h[1]^2), lap_bias, pad=0);
    mass_conv = Conv(mass_stencil, mass_bias, pad=0);
    
    x_repeat = pad_repeat(x,div(size(laplacian_stencil,1),2))
    mass_x_repeat = pad_repeat(mass.*x,div(size(mass_stencil,1),2))
    
    return laplacian_conv(x_repeat) .+ mass_conv(mass_x_repeat)
    # return laplacian_conv(real(x_repeat)) .+ im*laplacian_conv(imag(x_repeat)) .+ mass_conv(real(mass_x_repeat)) .+ im*mass_conv(imag(mass_x_repeat))
end