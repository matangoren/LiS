function getHelmholtzMatrices(m, omega, gamma, h; alpha=0.1)

    sommerfeld = zeros(ComplexF64, size(m))
    sommerfeld[1,:]  .+= (-1/h[1]^2) .+ ((1/h[1]) .* im*omega*sqrt.(m[1,:]))
    sommerfeld[end,:]  .+= (-1/h[1]^2) .+ ((1/h[1]) .* im*omega*sqrt.(m[end,:]))
    sommerfeld[:,1]  .+= (-1/h[2]^2) .+  ((1/h[2]) .* im*omega*sqrt.(m[:,1]))
    sommerfeld[:,end] .+= (-1/h[2]^2) .+ ((1/h[2]) .* im*omega*sqrt.(m[:,end]))

    helmholtz_matrix = (omega.^2).*m.*(1.0.-1im.*gamma/omega) .- sommerfeld
    sl_matrix = (omega.^2).*m.*(1.0.-1im.*(gamma/omega .+ alpha)) .- sommerfeld


    return sl_matrix, helmholtz_matrix
end

function getInterp(dim; highOrder=false)
    s = 0.5 * [1,2,1]
    pad = 1
    if highOrder
        s = (1/8) * [1,4,6,4,1]
        pad = 2
    end
    k = Int64.(length(s) .* ones(dim))
    P = copy(s)

    for i=2:dim
        P = kron(s,P)
    end
    P = Float64.(reshape(P,k...,1,1))
    R = P ./ 2^dim

    R = Conv(R, zeros(Float64,1), stride=2,pad=pad)
    P = ConvTranspose(P, zeros(Float64, 1), stride=2,pad=pad)
    return R,P
end