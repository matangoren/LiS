using DelimitedFiles

# Headless execution (remote server)
ENV["MPLBACKEND"] = "Agg"

using PyPlot

function plot_3d_slices(dat_file, save_path;
                        dims=(257, 257, 97),
                        cmap="jet",
                        clim=nothing,
                        colorbar_ticks=nothing)

    # -----------------------------------
    # Load 3D array
    # -----------------------------------
    A = readdlm(dat_file)
    A = reshape(Float64.(A), dims)

    nx, ny, nz = size(A)

    # Middle indices
    mx = div(nx + 1, 2)
    my = div(ny + 1, 2)
    mz = div(nz + 1, 2)

    # Coordinates
    x = collect(1:nx)
    y = collect(1:ny)
    z = collect(1:nz)

    # -----------------------------------
    # Color limits
    # -----------------------------------
    if clim === nothing
        vmin, vmax = extrema(A)
    else
        vmin, vmax = clim
    end

    if vmin >= vmax
        error("Color limits must satisfy vmin < vmax.")
    end

    # -----------------------------------
    # Slice 1: fixed z (XY plane)
    # -----------------------------------
    X1 = repeat(reshape(Float64.(x), 1, nx), ny, 1)
    Y1 = repeat(reshape(Float64.(y), ny, 1), 1, nx)
    Z1 = fill(Float64(mz), ny, nx)

    V1 = permutedims(A[:, :, mz])

    # -----------------------------------
    # Slice 2: fixed y (XZ plane)
    # -----------------------------------
    X2 = repeat(reshape(Float64.(x), 1, nx), nz, 1)
    Z2 = repeat(reshape(Float64.(z), nz, 1), 1, nx)
    Y2 = fill(Float64(my), nz, nx)

    V2 = permutedims(A[:, my, :])

    # -----------------------------------
    # Slice 3: fixed x (YZ plane)
    # -----------------------------------
    Y3 = repeat(reshape(Float64.(y), 1, ny), nz, 1)
    Z3 = repeat(reshape(Float64.(z), nz, 1), 1, ny)
    X3 = fill(Float64(mx), nz, ny)

    V3 = permutedims(A[mx, :, :])

    # -----------------------------------
    # Create figure
    # -----------------------------------
    fig = figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Common color normalization
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    color_map = PyPlot.get_cmap(cmap)

    # -----------------------------------
    # Plot three intersecting surfaces
    # -----------------------------------
    ax.plot_surface(
        X1, Y1, Z1;
        facecolors=color_map(norm.(V1)),
        shade=false,
        linewidth=0,
        antialiased=true
    )

    ax.plot_surface(
        X2, Y2, Z2;
        facecolors=color_map(norm.(V2)),
        shade=false,
        linewidth=0,
        antialiased=true
    )

    ax.plot_surface(
        X3, Y3, Z3;
        facecolors=color_map(norm.(V3)),
        shade=false,
        linewidth=0,
        antialiased=true
    )

    # -----------------------------------
    # Colorbar ticks
    # -----------------------------------
    if colorbar_ticks === nothing
        colorbar_ticks = range(vmin, vmax, length=5)
    end

    sm = matplotlib.cm.ScalarMappable(
        norm=norm,
        cmap=color_map
    )

    sm.set_array(Float64[])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        shrink=0.7,
        pad=0.1,
        ticks=colorbar_ticks
    )

    # -----------------------------------
    # Appearance
    # -----------------------------------
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")

    ax.set_xlim(1, nx)
    ax.set_ylim(1, ny)
    ax.set_zlim(1, nz)

    # Preserve physical proportions
    ax.set_box_aspect((nx, ny, nz))

    # Viewing angle
    ax.view_init(elev=25, azim=-55)

    ax.grid(false)

    # -----------------------------------
    # Save figure
    # -----------------------------------
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    close(fig)

    println("Figure saved to: ", save_path)

    return nothing
end


file = "$(@__DIR__)/Results/forPaper/LiSvsCompactLiS/CPU/tunnel/x_sol_257_257_97_LiS_1_5.dat"
output_file = "$(@__DIR__)/Results/forPaper/LiSvsCompactLiS/CPU/tunnel/x_sol_257_257_97_LiS_1_5_intersecting.png"
plot_3d_slices(
    file,
    output_file,
    dims=(257, 257, 97),
    cmap="jet",
    clim=(5*10^-13, 5*10^-6))