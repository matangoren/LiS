using PyPlot
using Statistics

function plot_method_performance(
    output_folder;
    filename = "method_performance.png"
)

    # --------------------------------------------------
    # Synthetic data
    # --------------------------------------------------

    methods = ["GMRES(20,20)", "GMRES(1,20)"]#, "GMRES(1,10)"]
    tolerances = [0.1,0.2,0.3,0.4,0.5]

    # Solve times in seconds.
    timings = Dict(
        methods[1] => [1730.17, 1603.33, 1255.99, 978.54, 736.27],
        methods[2] => [982.41, 990.92, 988.80,952.41, 1192.19],
        # methods[3] => [2356.96, 3033.47, 2256.96, 500,900],
    )

    # Iteration counts.
    iterations = Dict(
        methods[1] => [11, 11, 11, 12, 15],
        methods[2] => [105, 105, 105, 101, 127],
        # methods[3] => [300, 62, 85,140,300],
    )

    # --------------------------------------------------
    # Visual styles
    # --------------------------------------------------

    colors = Dict(
        methods[1] => "tab:blue",
        methods[2] => "tab:orange",
        # methods[3] => "tab:green",
    )

    markers = Dict(
        0.1 => "o",
        0.2 => "s",
        0.3 => "^",
        0.4 => "*",
        0.5 => "D",
    )

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------

    fig, ax = subplots(figsize = (11, 7))

    # --------------------------------------------------
    # Plot methods
    # X-axis: iterations
    # Y-axis: solve time
    # --------------------------------------------------

    for method in methods

        x = iterations[method]
        y = timings[method]

        # Connect points belonging to the same method.
        # ax.plot(
        #     x,
        #     y;
        #     color = colors[method],
        #     linewidth = 2,
        #     alpha = 0.85,
        #     zorder = 1,
        # )

        # Plot each tolerance using its marker shape.
        for (j, tol) in enumerate(tolerances)

            ax.scatter(
                x[j],
                y[j];
                color = colors[method],
                marker = markers[tol],
                s = 200,
                edgecolors = "black",
                linewidths = 0.8,
                zorder = 3,
            )
        end
    end

    # --------------------------------------------------
    # Axes and title
    # --------------------------------------------------

    ax.set_xlabel("Iteration count", fontsize = 13)
    ax.set_ylabel("Solve time (seconds)", fontsize = 13)

    ax.set_title(
        "Method Performance Comparison",
        fontsize = 15,
        pad = 12,
    )

    ax.grid(
        true;
        linestyle = "--",
        alpha = 0.35,
    )

    # --------------------------------------------------
    # Method legend
    # --------------------------------------------------

    method_handles = [
        ax.plot(
            Float64[],
            Float64[];
            color = colors[method],
            linewidth = 2.5,
            label = method,
        )[1]
        for method in methods
    ]

    legend_methods = ax.legend(
                    handles = method_handles,
                    title = "Method",
                    loc = "lower right",
                    frameon = true,
                    fontsize = 11,
                    title_fontsize = 12,
                )

    # --------------------------------------------------
    # Tolerance legend
    # Place close below the method legend.
    # --------------------------------------------------

    tolerance_handles = [
        ax.scatter(
            Float64[],
            Float64[];
            color = "gray",
            marker = markers[tol],
            s = 100,
            edgecolors = "black",
            label = "tol = $tol",
        )
        for tol in tolerances
    ]

    ax.legend(
        handles = tolerance_handles,
        title = "Tolerance",
        loc = "lower left",
        bbox_to_anchor = (1.03, 0.0),
        borderaxespad = 0,
        frameon = true,
        fontsize = 11,
        title_fontsize = 12,
    )

    # Preserve both legends.
    ax.add_artist(legend_methods)

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    fig.subplots_adjust(
        left = 0.12,
        right = 0.73,
        bottom = 0.13,
        top = 0.90,
    )

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------

    mkpath(output_folder)

    output_path = joinpath(output_folder, filename)

    savefig(
        output_path;
        dpi = 300,
        bbox_inches = "tight",
    )

    println("Plot saved to: $output_path")

    close(fig)

    return output_path
end


# --------------------------------------------------
# Run
# --------------------------------------------------

output_folder = joinpath(@__DIR__, "dump")

plot_method_performance(output_folder)
