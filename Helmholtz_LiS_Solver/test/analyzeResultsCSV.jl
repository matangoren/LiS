using CSV
using DataFrames
using Statistics
using Printf


# ============================================================
# Collect results from CSV files
# ============================================================

function collect_results(results_folder)

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
        # cpu_1025_257_4_16_marmousi_GMRES_Kmeans4_tol-0.1.csv
        # cpu_1025_257_4_16_marmousi_LiS_Kmeans5_tol-0.1.csv
        # cpu_1025_257_4_16_marmousi_compact_LiS_Kmeans6_tol-0.1.csv

        m = match(
            r"cpu_(\d+)_(\d+)_4_16_marmousi_(.+)_Kmeans(\d+)_tol-([\d.]+)\.csv",
            filename
        )

        if m === nothing
            println("Skipping file with unexpected name: $filename")
            continue
        end

        nx = parse(Int, m.captures[1])
        ny = parse(Int, m.captures[2])
        solver = m.captures[3]
        kmeans = parse(Int, m.captures[4])
        tol = parse(Float64, m.captures[5])

        grid_size = "$(ny) × $(nx)"
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
            # Keep the first GMRES result if duplicates exist.

            if !haskey(results[tol][grid_size], "GMRES")
                results[tol][grid_size]["GMRES"] = statistics
            end

        elseif solver == "LiS" || solver == "compact_LiS"

            if kmeans in (4, 5, 6)

                solver_key = if solver == "LiS"
                    "LiS_k$(kmeans)"
                else
                    "compact_LiS_k$(kmeans)"
                end

                results[tol][grid_size][solver_key] = statistics

            else

                println(
                    "Unsupported Kmeans value $kmeans in file: $filename"
                )

            end

        else

            println(
                "Unknown solver '$solver' in file: $filename"
            )

        end

    end

    return results
end


# ============================================================
# Print tables to terminal AND TXT file
# ============================================================

function print_tables(results; output_file=nothing)

    # --------------------------------------------------------
    # Solver keys and displayed headers
    # --------------------------------------------------------

    solvers = [
        "GMRES",
        "LiS_k4",
        "LiS_k5",
        "LiS_k6",
        "compact_LiS_k4",
        "compact_LiS_k5",
        "compact_LiS_k6"
    ]

    headers = [
        "GMRES",
        "LiS (k=4)",
        "LiS (k=5)",
        "LiS (k=6)",
        "compact LiS (k=4)",
        "compact LiS (k=5)",
        "compact LiS (k=6)"
    ]

    # --------------------------------------------------------
    # Column widths
    #
    # Literal format widths are used for Julia 1.7.3.
    # Do not use "%-*s", which is unsupported.
    # --------------------------------------------------------

    grid_width = 18
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
    # Print to both destinations
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
        print_both("SOLVER COMPARISON TABLES")
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
                "%-18s | ",
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
            # Sort grid sizes by total grid points
            # ------------------------------------------------

            grids = collect(keys(results[tol]))

            sort!(
                grids,
                by = grid -> begin
                    parts = split(grid, " × ")
                    ny = parse(Int, parts[1])
                    nx = parse(Int, parts[2])
                    ny * nx
                end
            )

            # =================================================
            # Print each grid
            # =================================================

            for grid in grids

                # First line: iterations
                row_iter = @sprintf(
                    "%-18s | ",
                    grid
                )

                # Second line: time
                row_time = @sprintf(
                    "%-18s | ",
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

                    # Add values using fixed-width formats
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

function create_solver_comparison(results_folder, output_folder)

    mkpath(output_folder)

    results = collect_results(results_folder)

    output_file = joinpath(
        output_folder,
        "solver_comparison.txt"
    )

    print_tables(
        results;
        output_file=output_file
    )

    return results
end

results_folder = "$(@__DIR__)/Results/marmousi/CPU/tol_study"
output_folder = "$(@__DIR__)/Results/marmousi/CPU/tol_study_summary"

results = create_solver_comparison(
    results_folder,
    output_folder
)

