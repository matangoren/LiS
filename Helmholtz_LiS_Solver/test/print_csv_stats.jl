using CSV
using DataFrames
using Statistics
using Printf

function summarize_csv_files(input_dir, output_file)

    # Find all CSV files in the folder.
    csv_files = sort(filter(
        filename -> endswith(lowercase(filename), ".csv"),
        readdir(input_dir)
    ))

    open(output_file, "w") do io

        for filename in csv_files

            filepath = joinpath(input_dir, filename)

            # Read CSV without headers.
            df = CSV.read(filepath, DataFrame; header=false)

            rename!(
                df,
                [
                    :coarse_run_time,
                    :coarse_iteration_count,
                    :coarse_error
                ]
            )

            # Exclude rows with zero error before calculating statistics.
            df = filter(row -> row.coarse_error != 0, df)

            println(io, filename, ":")

            if nrow(df) == 0
                println(io, "  No rows remaining after filtering.")
                println(io)
                continue
            end

            # Extract filtered data.
            iterations = df.coarse_iteration_count
            times = df.coarse_run_time

            # Calculate statistics.
            total_iterations = sum(iterations)
            mean_iterations = mean(iterations)
            std_iterations = std(iterations)

            total_time = sum(times)
            mean_time = mean(times)
            std_time = std(times)

            # Write statistics.
            println(io, "  Total iterations: ", total_iterations)
            println(
                io,
                "  Mean iterations: ",
                @sprintf("%.6f", mean_iterations)
            )
            println(
                io,
                "  Std iterations: ",
                @sprintf("%.6f", std_iterations)
            )

            println(io, "  Total time: ", @sprintf("%.6f", total_time))
            println(
                io,
                "  Mean time: ",
                @sprintf("%.6f", mean_time)
            )
            println(
                io,
                "  Std time: ",
                @sprintf("%.6f", std_time)
            )

            # Blank line between files.
            println(io)
        end
    end

    println("Wrote summary: ", output_file)

    return nothing
end


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

for model in ["linear", "marmousi"]
    input_dir = "$(@__DIR__)/Results/forPaper/LiSvsCompactLiS/CPU/$(model)"
    output_file = "$(@__DIR__)/Results/forPaper/LiSvsCompactLiS/CPU/LiSvsCompactLiS_$(model)_summary.txt"

    summarize_csv_files(input_dir, output_file)
end
