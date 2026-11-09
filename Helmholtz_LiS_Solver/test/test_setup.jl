T = "High"
orderNeumannBC = 1
add_sommerfeld = true


fine_laplacian_type = T
fine_mass_type = T

level = 3

# dim = 2
# dim is given from the command line arguments

if dim == 2
    laplacian_types = laplacian_types_2D
    mass_types = mass_types_2D
else
    laplacian_types = laplacian_types_3D
    mass_types = mass_types_3D
end

fine_laplacian_stencil = laplacian_types[fine_laplacian_type]
fine_mass_stencil = mass_types[fine_mass_type]

shift = 0.0
if T == "Low"
    if level == 3
        shift = 0.3
    elseif level == 4
        shift = 0.5
    end
else
    if level == 3
        shift = 0.1
    elseif level == 4
        shift = 0.3
    end
end
alphas = ones(data_type,level) * shift
