using CSV
using DataFrames
using Printf
using Statistics

function summarize_coarse_results(input_dir, output_dir, filename, totals, pattern)

    mkpath(output_dir)

    
    results = Dict{String, Vector{NamedTuple}}()

    for filename in readdir(input_dir)

        endswith(lowercase(filename), ".csv") || continue

        match_result = match(pattern, filename)
        isnothing(match_result) && continue

        grid_size = match_result.captures[1]
        solver = match_result.captures[2]
        tol = string(match_result.captures[3])

        key = (solver, tol, grid_size)

        # If the totals dictionary has no entry, record N/A.
        # As in the original code, do not read the CSV in this case.
        if !haskey(totals, key)

            @warn "Missing dictionary key; writing N/A" key filename

            record = (
                grid_size = grid_size,
                solver = solver,
                total_time = missing,
                total_iterations = missing,
                coarse_iterations = missing,
                mean_coarse_iterations = missing,
                std_coarse_iterations = missing,
                coarse_time = missing,
                mean_coarse_time = missing,
                std_coarse_time = missing
            )

            push!(get!(results, tol, NamedTuple[]), record)
            continue
        end

        # Read CSV without headers.
        filepath = joinpath(input_dir, filename)
        df = CSV.read(filepath, DataFrame; header=false)

        rename!(
            df,
            [
                :coarse_run_time,
                :coarse_iteration_count,
                :coarse_error
            ]
        )

        # IMPORTANT: filter before calculating every CSV statistic.
        # Exclude rows whose coarse_error is exactly zero.
        df_filtered = filter(row -> row.coarse_error != 0, df)

        # Total statistics from the dictionary.
        total_iterations, total_time = totals[key]

        # CSV statistics, computed only from filtered rows.
        if nrow(df_filtered) > 0

            coarse_iterations =
                sum(df_filtered.coarse_iteration_count)

            mean_coarse_iterations =
                mean(df_filtered.coarse_iteration_count)

            std_coarse_iterations =
                std(df_filtered.coarse_iteration_count)

            coarse_time =
                sum(df_filtered.coarse_run_time)

            mean_coarse_time =
                mean(df_filtered.coarse_run_time)

            std_coarse_time =
                std(df_filtered.coarse_run_time)

        else
            # No rows remain after filtering.
            coarse_iterations = missing
            mean_coarse_iterations = missing
            std_coarse_iterations = missing

            coarse_time = missing
            mean_coarse_time = missing
            std_coarse_time = missing
        end

        record = (
            grid_size = grid_size,
            solver = solver,
            total_time = total_time,
            total_iterations = total_iterations,
            coarse_iterations = coarse_iterations,
            mean_coarse_iterations = mean_coarse_iterations,
            std_coarse_iterations = std_coarse_iterations,
            coarse_time = coarse_time,
            mean_coarse_time = mean_coarse_time,
            std_coarse_time = std_coarse_time
        )

        push!(get!(results, tol, NamedTuple[]), record)

    end

    output_file = joinpath(output_dir, filename)

    # Format iteration counts.
    function format_iterations(x)
        if ismissing(x)
            return "N/A"
        elseif x > 300
            return ">300"
        else
            return string(x)
        end
    end

    # Format time and other numerical values.
    function format_value(x)
        ismissing(x) && return "N/A"
        return @sprintf("%.6f", x)
    end

    # Format mean +/- standard deviation.
    function format_mean_std(mu, sigma)
        if ismissing(mu) || ismissing(sigma)
            return "N/A"
        end

        return @sprintf("%.6f +/- %.6f", mu, sigma)
    end

    open(output_file, "w") do io

        tolerances = sort(collect(keys(results)))

        for (table_idx, tol) in enumerate(tolerances)

            records = results[tol]

            grid_sizes = sort(unique(r.grid_size for r in records))
            solvers = sort(unique(r.solver for r in records))

            lookup = Dict(
                (r.grid_size, r.solver) => r for r in records
            )

            headers = ["Grid size / metric"; solvers]
            table_rows = Vector{Vector{String}}()

            for grid in grid_sizes

                # Each grid has six rows, one per requested statistic.
                metrics = [
                    (
                        "Total iterations",
                        r -> format_iterations(r.total_iterations)
                    ),
                    (
                        "Total solve time",
                        r -> format_value(r.total_time)
                    ),
                    (
                        "Coarse iterations (sum)",
                        r -> format_iterations(r.coarse_iterations)
                    ),
                    (
                        "Coarse iterations (mean +/- std)",
                        r -> format_mean_std(
                            r.mean_coarse_iterations,
                            r.std_coarse_iterations
                        )
                    ),
                    (
                        "Coarse time (sum)",
                        r -> format_value(r.coarse_time)
                    ),
                    (
                        "Coarse time (mean +/- std)",
                        r -> format_mean_std(
                            r.mean_coarse_time,
                            r.std_coarse_time
                        )
                    )
                ]

                for (metric_name, get_value) in metrics

                    row = String["$grid ($metric_name)"]

                    for solver in solvers

                        if haskey(lookup, (grid, solver))
                            r = lookup[(grid, solver)]
                            push!(row, get_value(r))
                        else
                            push!(row, "N/A")
                        end

                    end

                    push!(table_rows, row)
                end

            end

            # Determine column widths from the longest entry.
            widths = [
                maximum(
                    length(row[j])
                    for row in vcat([headers], table_rows)
                ) + 2
                for j in eachindex(headers)
            ]

            # Table title.
            println(io, "tol = ", tol)

            # Separator.
            separator = join(["-"^w for w in widths], "+")

            # Header.
            println(
                io,
                join(
                    [
                        rpad(headers[j], widths[j])
                        for j in eachindex(headers)
                    ],
                    "|"
                )
            )

            println(io, separator)

            # Aligned table rows.
            for row in table_rows

                print(io, rpad(row[1], widths[1]), "|")

                for j in 2:length(row)
                    print(
                        io,
                        lpad(row[j], widths[j]),
                        j < length(row) ? "|" : ""
                    )
                end

                println(io)
            end

            # Five blank lines between tolerance tables.
            if table_idx < length(tolerances)
                println(io, "\n\n\n\n\n")
            end

        end
    end

    println("Wrote summary: ", output_file)

    return nothing
end


# totals = Dict(
#     ("GMRES_1_10", "0.1", "33_33_33") => (8,0.39),
#     ("GMRES_1_10", "0.1", "65_65_65") => (9,3.51),
#     ("GMRES_1_10", "0.1", "129_129_129") => (13,67.33),
#     ("GMRES_1_10", "0.1", "257_257_257") => (21,805.38),

#     ("GMRES_1_10", "0.2", "33_33_33") => (8,0.44),
#     ("GMRES_1_10", "0.2", "65_65_65") => (9,3.78),
#     ("GMRES_1_10", "0.2", "129_129_129") => (13,70.77),
#     ("GMRES_1_10", "0.2", "257_257_257") => (21,813.26),

#     ("GMRES_1_10", "0.3", "33_33_33") => (8,0.42),
#     ("GMRES_1_10", "0.3", "65_65_65") => (10,3.99),
#     ("GMRES_1_10", "0.3", "129_129_129") => (13,70.69),
#     ("GMRES_1_10", "0.3", "257_257_257") => (21,808.67),

#     ("GMRES_1_10", "0.4", "33_33_33") => (10,0.48),
#     ("GMRES_1_10", "0.4", "65_65_65") => (11,4.46),
#     ("GMRES_1_10", "0.4", "129_129_129") => (14,75.80),
#     ("GMRES_1_10", "0.4", "257_257_257") => (22,846.73),

#     ("GMRES_1_10", "0.5", "33_33_33") => (12,0.52),
#     ("GMRES_1_10", "0.5", "65_65_65") => (20,6.65),
#     ("GMRES_1_10", "0.5", "129_129_129") => (20,108.79),
#     ("GMRES_1_10", "0.5", "257_257_257") => (27,999.41),

#     ("compactLiS_1_5", "0.1", "33_33_33") => (8,0.39),
#     ("compactLiS_1_5", "0.1", "65_65_65") => (9,3.12),
#     ("compactLiS_1_5", "0.1", "129_129_129") => (10,48.25),
#     ("compactLiS_1_5", "0.1", "257_257_257") => (14,526.40),

#     ("compactLiS_1_5", "0.2", "33_33_33") => (8,0.44),
#     ("compactLiS_1_5", "0.2", "65_65_65") => (9,3.28),
#     ("compactLiS_1_5", "0.2", "129_129_129") => (11,65.14),
#     ("compactLiS_1_5", "0.2", "257_257_257") => (14,524.33),

#     ("compactLiS_1_5", "0.3", "33_33_33") => (9,0.447),
#     ("compactLiS_1_5", "0.3", "65_65_65") => (11,4.32),
#     ("compactLiS_1_5", "0.3", "129_129_129") => (14,81.66),
#     ("compactLiS_1_5", "0.3", "257_257_257") => (17,613.60),

#     ("compactLiS_1_5", "0.4", "33_33_33") => (11,0.59),
#     ("compactLiS_1_5", "0.4", "65_65_65") => (11,3.93),
#     ("compactLiS_1_5", "0.4", "129_129_129") => (15,91.12),
#     ("compactLiS_1_5", "0.4", "257_257_257") => (21,762.69),

#     ("compactLiS_1_5", "0.5", "33_33_33") => (12,0.60),
#     ("compactLiS_1_5", "0.5", "65_65_65") => (33,10.21),
#     ("compactLiS_1_5", "0.5", "129_129_129") => (255,1294.89),
#     ("compactLiS_1_5", "0.5", "257_257_257") => (201,7035.82),
#     )



input_dir = "$(@__DIR__)/Results/forPaper/DDcomparison/CPU/linear"
output_dir = "$(@__DIR__)/Results/forPaper/DDcomparison/CPU"
filename = "DDcomparison_linear_summary.txt"

# Expected filename:
# cpu_<grid>_1_1_1_linear_<solver>_ExactKmeans4_tol-<tol>.csv
pattern = r"^cpu_(.+?)_DD_(.+?)_linear_(.+?)_tol-([0-9.eE+-]+)\.csv$"


summarize_coarse_results(input_dir, output_dir, filename, totals, pattern)
