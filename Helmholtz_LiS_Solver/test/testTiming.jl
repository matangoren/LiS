include("./test_import.jl")

include("../src/solvers/LiS_solver.jl")
include("../src/solvers/multigrid_solver.jl")

using PyPlot


# ============================================================
# Timing functions
# ============================================================

function getLiSTiming(lis_solver, b; verbose=true)

    lis_total_time = 0.0

    for i = 1:10

        if CUDA.functional()
            CUDA.synchronize()
        end

        t0 = time_ns()

        x = weighted_LiS_solve(lis_solver, b)

        if CUDA.functional()
            CUDA.synchronize()
        end

        t = (time_ns() - t0) / 1e9
        lis_total_time += t
    end

    # Average over the 10 repetitions
    lis_time = lis_total_time / 10

    if verbose
        println("\t -> LiS Time - $(size(lis_solver.T_LiS)): $(lis_time)")
    end

    return lis_time
end


function getJacobiTiming(
    h,
    laplacian_stencil_forH,
    mass_stencil_forH,
    b;
    verbose=true
)

    jacobi_total_time = 0.0

    x = reshape(b, size(b)..., 1, 1)

    for i = 1:10

        if CUDA.functional()
            CUDA.synchronize()
        end

        t0 = time_ns()

        temp = helmholtzJacobi(
            x,
            x,
            h,
            x,
            x,
            laplacian_stencil_forH,
            mass_stencil_forH;
            max_iter=1
        )

        if CUDA.functional()
            CUDA.synchronize()
        end

        t = (time_ns() - t0) / 1e9
        jacobi_total_time += t
    end

    # Average over the 10 repetitions
    jacobi_time = jacobi_total_time / 10

    if verbose
        println("\t -> Jacobi Time: $(jacobi_time)")
    end

    return jacobi_time
end


# ============================================================
# Measure all timings
# ============================================================

function measureTimes(; dim=2)

    if dim == 2
        stencil_shape = (7, 7)
        ns = [64, 128, 256]

    else
        stencil_shape = (7, 7, 7)
        ns = [32, 64, 128]
    end

    # --------------------------------------------------------
    # Stencils
    # --------------------------------------------------------

    laplacian_stencil = ones(Float64, stencil_shape) |> device
    mass_stencil = laplacian_stencil

    laplacian_stencil_forH = reshape(laplacian_stencil, size(laplacian_stencil)..., 1, 1)

    mass_stencil_forH = laplacian_stencil_forH

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_timing_jacobi = Float64[]

    # Each entry contains:
    #   [time for 3-term LiS,
    #    time for 4-term LiS,
    #    time for 5-term LiS]
    all_timing_lis = Vector{Vector{Float64}}()

    # --------------------------------------------------------
    # Loop over grid sizes
    # --------------------------------------------------------

    for i in ns

        n = fill(i + 1, dim)

        println("n = $(n)")

        h = 1 ./ n

        # ----------------------------------------------------
        # Delta
        # ----------------------------------------------------

        delta = zeros(ComplexF64, n...)

        if dim == 2

            delta[
                div(n[1], 2) + 2,
                div(n[2], 2) + 2
            ] = 1.0

        else

            delta[
                div(n[1], 2) + 2,
                div(n[2], 2) + 2,
                div(n[3], 2) + 2
            ] = 1.0

        end

        delta = ComplexF64.(delta |> device)

        # ----------------------------------------------------
        # RHS
        # ----------------------------------------------------

        b = ones(ComplexF64, n...) |> device

        omega = 1.0

        # ----------------------------------------------------
        # Jacobi
        # ----------------------------------------------------
        temp = getJacobiTiming(
            h,
            laplacian_stencil_forH,
            mass_stencil_forH,
            b;
            verbose=false
        )
        jacobi_timing = getJacobiTiming(
            h,
            laplacian_stencil_forH,
            mass_stencil_forH,
            b;
            verbose=true
        )

        push!(all_timing_jacobi, jacobi_timing)

        # ----------------------------------------------------
        # LiS solvers
        # num_lis = 3, 4, 5
        # ----------------------------------------------------

        n_lis_timing = Float64[]

        for num_lis = 3:5

            vals = fill(0.5, num_lis)

            m0_s = ComplexF64.(
                (omega)^2 .* vals .* (1 - im * 0.01)
            )

            lis_solver = getLiS_solver(
                laplacian_stencil,
                mass_stencil,
                n,
                h,
                delta,
                m0_s,
                4 * ones(Int64, dim),
                4 * ones(Int64, dim)
            )

            # ------------------------------------------------
            # Test T_LiS
            # ------------------------------------------------
            lis_solver.T_LiS = [
                ones(ComplexF64, n...) |> device
                for i = 1:length(vals)
            ]
            
            temp = getLiSTiming(
                lis_solver,
                b;
                verbose=false
            )

            lis_timing = getLiSTiming(
                lis_solver,
                b;
                verbose=true
            )

            push!(n_lis_timing, lis_timing)
        end

        push!(all_timing_lis, n_lis_timing)

        println()
    end

    return ns, all_timing_jacobi, all_timing_lis
end


# ============================================================
# Plot timings
# ============================================================

function plotTimings(
    ns,
    all_timing_jacobi,
    all_timing_lis;
    output_file="timings.png"
)

    figure(figsize=(10, 7))

    # --------------------------------------------------------
    # Bar positions
    # --------------------------------------------------------

    num_n = length(ns)
    x = collect(1:num_n)

    bar_width = 0.18

    # Four bars per grid size:
    #
    #        Jacobi   LiS3   LiS4   LiS5
    #           |       |      |      |
    #
    #        ---- group centered around x ----

    jacobi_positions = x .- 1.5 .* bar_width
    lis3_positions   = x .- 0.5 .* bar_width
    lis4_positions   = x .+ 0.5 .* bar_width
    lis5_positions   = x .+ 1.5 .* bar_width

    # --------------------------------------------------------
    # Jacobi
    # --------------------------------------------------------

    bar(
        jacobi_positions,
        all_timing_jacobi,
        width=bar_width,
        label="Jacobi"
    )

    # --------------------------------------------------------
    # LiS 3
    # --------------------------------------------------------

    lis_times = [
        all_timing_lis[i][1]
        for i = 1:num_n
    ]

    bar(
        lis3_positions,
        lis_times,
        width=bar_width,
        label="LiS (3)"
    )

    for i = 1:num_n

        ratio = all_timing_jacobi[i] / all_timing_lis[i][1]

        text(
            lis3_positions[i],
            lis_times[i] * 1.03,
            "x$(round(ratio, digits=2))",
            ha="center",
            va="bottom",
            fontsize=9
        )
    end

    # --------------------------------------------------------
    # LiS 4
    # --------------------------------------------------------

    lis_times = [
        all_timing_lis[i][2]
        for i = 1:num_n
    ]

    bar(
        lis4_positions,
        lis_times,
        width=bar_width,
        label="LiS (4)"
    )

    for i = 1:num_n

        ratio = all_timing_jacobi[i] / all_timing_lis[i][2]

        text(
            lis4_positions[i],
            lis_times[i] * 1.03,
            "x$(round(ratio, digits=2))",
            ha="center",
            va="bottom",
            fontsize=9
        )
    end

    # --------------------------------------------------------
    # LiS 5
    # --------------------------------------------------------

    lis_times = [
        all_timing_lis[i][3]
        for i = 1:num_n
    ]

    bar(
        lis5_positions,
        lis_times,
        width=bar_width,
        label="LiS (5)"
    )

    for i = 1:num_n

        ratio = all_timing_jacobi[i] / all_timing_lis[i][3]

        text(
            lis5_positions[i],
            lis_times[i] * 1.03,
            "x$(round(ratio, digits=2))",
            ha="center",
            va="bottom",
            fontsize=9
        )
    end

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    xlabel("Grid size (n)", fontsize=13)
    ylabel("Time [s]", fontsize=13)

    title(
        "Jacobi vs. LiS Solver Timing",
        fontsize=14
    )

    xticks(
        x,
        string.(ns)
    )

    grid(
        true,
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    legend(
        fontsize=11,
        loc="best"
    )

    tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    println("Figure saved to: $(output_file)")

    show()
end


# ============================================================
# Run
# ============================================================

ns, all_timing_jacobi, all_timing_lis = measureTimes(dim=2)

plotTimings(
    ns,
    all_timing_jacobi,
    all_timing_lis;
    output_file="$(@__DIR__)/dump/timings_2D_GPU.png"
)

ns, all_timing_jacobi, all_timing_lis = measureTimes(dim=3)

plotTimings(
    ns,
    all_timing_jacobi,
    all_timing_lis;
    output_file="$(@__DIR__)/dump/timings_3D_GPU.png"
)