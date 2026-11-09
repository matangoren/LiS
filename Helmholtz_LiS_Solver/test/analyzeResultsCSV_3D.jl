
using CSV
using DataFrames
using Statistics
using Printf


# ============================================================
# Collect results from 3D CSV files
# ============================================================

function collect_results_3D(results_folder)

    results = Dict{
        Float64,
        Dict{
            String,
            Dict{
                String,
                Tuple{Float64, Float64, Float64, Float64}
            }
        }
    }()

    for filename in readdir(results_folder)

        # Process CSV files only
        if !endswith(lowercase(filename), ".csv")
            continue
        end

        # Expected filename examples:
        #
        # cpu_65_65_65_1_1_1_linear_GMRES_Kmeans4_tol-0.1.csv
        # cpu_65_65_65_1_1_1_linear_LiS_Kmeans4_tol-0.1.csv
        # cpu_65_65_65_1_1_1_linear_compact_LiS_Kmeans4_tol-0.1.csv
        #
        # The first three numbers are the grid dimensions.
        # GMRES ignores the Kmeans value.

        m = match(
            r"cpu_(\d+)_(\d+)_(\d+)_\d+_\d+_\d+_linear_(.+)_Kmeans(\d+)_tol-([\d.]+)\.csv",
            filename
        )

        if m === nothing
            println("Skipping file with unexpected name: $filename")
            continue
        end

        nx = parse(Int, m.captures[1])
        ny = parse(Int, m.captures[2])
        nz = parse(Int, m.captures[3])

        solver = m.captures[4]
        kmeans = parse(Int, m.captures[5])
        tol = parse(Float64, m.captures[6])

        # Display dimensions in the filename's order
        grid_size = "$(nx) × $(ny) × $(nz)"

        filepath = joinpath(results_folder, filename)

        # ----------------------------------------------------
        # Read CSV
        #
        # Column 1: time
        # Column 2: iteration count
        # Column 3: error norm
        # ----------------------------------------------------

        df = CSV.read(filepath, DataFrame)

        df = filter(row -> row[3] > 0, df)

        if nrow(df) == 0
            println("No valid rows in: $filename")
            continue
        end

        # ----------------------------------------------------
        # Calculate statistics
        # ----------------------------------------------------

        time_values = df[:, 1]
        iteration_values = df[:, 2]

        statistics = (
            mean(time_values),
            std(time_values),
            mean(iteration_values),
            std(iteration_values)
        )

        # ----------------------------------------------------
        # Initialize dictionary levels
        # ----------------------------------------------------

        if !haskey(results, tol)
            results[tol] = Dict()
        end

        if !haskey(results[tol], grid_size)
            results[tol][grid_size] = Dict()
        end

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        if solver == "GMRES"

            # Kmeans does not affect GMRES.
            # Keep the first GMRES result if duplicate files exist.

            if !haskey(results[tol][grid_size], "GMRES")
                results[tol][grid_size]["GMRES"] = statistics
            end

        elseif solver == "LiS"

            if kmeans == 4
                results[tol][grid_size]["LiS_k4"] = statistics
            else
                println(
                    "Skipping LiS with unsupported Kmeans=$kmeans: $filename"
                )
            end

        elseif solver == "compact_LiS"

            if kmeans == 4
                results[tol][grid_size]["compact_LiS_k4"] = statistics
            else
                println(
                    "Skipping compact_LiS with unsupported Kmeans=$kmeans: $filename"
                )
            end

        else

            println("Unknown solver '$solver' in file: $filename")

        end

    end

    return results
end


# ============================================================
# Print tables to terminal AND TXT file
# ============================================================

function print_tables_3D(results; output_file=nothing)

    solvers = [
        "GMRES",
        "LiS_k4",
        "compact_LiS_k4"
    ]

    headers = [
        "GMRES",
        "LiS (k=4)",
        "compact LiS (k=4)"
    ]

    # Fixed widths: compatible with Julia 1.7.3 Printf.
    # Avoid dynamic formats such as "%-*s".

    grid_width = 22
    solver_width = 30
    num_solvers = length(solvers)

    total_width =
        grid_width +
        num_solvers * solver_width +
        (num_solvers + 1) * 3

    io_file = nothing

    if output_file !== nothing
        io_file = open(output_file, "w")
    end

    # --------------------------------------------------------
    # Print to both terminal and file
    # --------------------------------------------------------

    function print_both(text="")

        println(text)

        if io_file !== nothing
            println(io_file, text)
        end

    end

    try

        print_both()
        print_both("=" ^ total_width)
        print_both("3D SOLVER COMPARISON TABLES")
        print_both("=" ^ total_width)

        # ====================================================
        # Loop over tolerances
        # ====================================================

        for tol in sort(collect(keys(results)))

            print_both()
            print_both("Tolerance = $tol")
            print_both("-" ^ total_width)

            # ------------------------------------------------
            # Header
            # ------------------------------------------------

            header = @sprintf(
                "%-22s | ",
                "Grid size"
            )

            for (i, name) in enumerate(headers)

                header *= @sprintf(
                    "%-30s",
                    name
                )

                if i < num_solvers
                    header *= " | "
                end

            end

            print_both(header)
            print_both("-" ^ total_width)

            # ------------------------------------------------
            # Sort grid sizes by number of grid points
            # ------------------------------------------------

            grids = collect(keys(results[tol]))

            sort!(
                grids,
                by = grid -> begin
                    parts = split(grid, " × ")
                    nx = parse(Int, parts[1])
                    ny = parse(Int, parts[2])
                    nz = parse(Int, parts[3])
                    nx * ny * nz
                end
            )

            # =================================================
            # Print each grid
            # =================================================

            for grid in grids

                # First line: iterations
                row_iter = @sprintf(
                    "%-22s | ",
                    grid
                )

                # Second line: time
                row_time = @sprintf(
                    "%-22s | ",
                    ""
                )

                for (i, solver) in enumerate(solvers)

                    if haskey(results[tol][grid], solver)

                        (
                            time_mean,
                            time_std,
                            iter_mean,
                            iter_std
                        ) = results[tol][grid][solver]

                        iter_value = @sprintf(
                            "Iter: %.2f ± %.2f",
                            iter_mean,
                            iter_std
                        )

                        time_value = @sprintf(
                            "Time: %.5f ± %.5f",
                            time_mean,
                            time_std
                        )

                    else

                        iter_value = "—"
                        time_value = ""

                    end

                    row_iter *= @sprintf(
                        "%-30s",
                        iter_value
                    )

                    row_time *= @sprintf(
                        "%-30s",
                        time_value
                    )

                    if i < num_solvers
                        row_iter *= " | "
                        row_time *= " | "
                    end

                end

                print_both(row_iter)
                print_both(row_time)

            end

            print_both("-" ^ total_width)
            print_both(
                "Format: Iterations mean ± std; Time mean ± std"
            )
            print_both()

        end

        print_both("=" ^ total_width)

    finally

        if io_file !== nothing
            close(io_file)
        end

    end

    if output_file !== nothing
        println()
        println("Text file created:")
        println(output_file)
    end

end


# ============================================================
# Main function
# ============================================================

function create_solver_comparison_3D(results_folder, output_folder)

    mkpath(output_folder)

    results = collect_results_3D(results_folder)

    output_file = joinpath(
        output_folder,
        "solver_comparison_3D.txt"
    )

    print_tables_3D(
        results;
        output_file=output_file
    )

    return results
end

results_folder = "$(@__DIR__)/Results/linear/CPU/tol_study"
output_folder = "$(@__DIR__)/Results/linear/CPU/tol_study_summary"

results = create_solver_comparison_3D(
    results_folder,
    output_folder
)