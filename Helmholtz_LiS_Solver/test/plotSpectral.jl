using PyPlot

# Use LaTeX-style fonts throughout the figure
rc("font", family="serif", size=11)
rc("mathtext", fontset="cm")
rc("axes", titlesize=11, labelsize=12)
rc("xtick", labelsize=10)
rc("ytick", labelsize=10)

function plot_spectral_radius(V1, V2; save_path="spectral_radius.png")
    @assert length(V1) == 10 && length(V2) == 10

    τ = 1:10

    # Publication-style colors
    blue = "#3568A8"
    teal = "#218C87"

    fig, axes = subplots(
        2, 1;
        figsize=(7.2, 5.2),
        sharex=true
    )

    # Linear model
    axes[1].plot(
        τ, V1;
        color=blue,
        linewidth=2.0,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=blue,
        markeredgewidth=1.6
    )

    axes[1].text(
        0.98, 0.08, raw"$\mathrm{Linear\ model}$";
        transform=axes[1].transAxes,
        ha="right", va="bottom",
        fontsize=11, color=blue
    )

    # Marmousi
    axes[2].plot(
        τ, V2;
        color=teal,
        linewidth=2.0,
        marker="s",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=teal,
        markeredgewidth=1.6
    )

    axes[2].text(
        0.98, 0.08, raw"$\mathrm{Marmousi}$";
        transform=axes[2].transAxes,
        ha="right", va="bottom",
        fontsize=11, color=teal
    )

    # Axis formatting
    for ax in axes
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(0, 2)
        ax.set_xticks(τ)
        ax.set_ylabel(raw"$\rho$")
        ax.grid(true, linestyle="--", linewidth=0.6, alpha=0.3)
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.tick_params(direction="out", length=3.5, width=0.8)
    end

    axes[2].set_xlabel(raw"$\tau$", fontsize=13)

    # Shared title, positioned close to the top subplot
    fig.suptitle(
        raw"Iteration matrix spectral radius with respect to $\tau$",
        fontsize=13,
        y=0.985
    )

    # Tight vertical spacing and room for the title
    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.12,
        top=0.91,
        hspace=0.12
    )

    # Save high-resolution figure
    savefig("$(save_path).png"; dpi=300, bbox_inches="tight")
    savefig("$(save_path).eps"; dpi=300, bbox_inches="tight")
    close(fig)

    println("Plot saved to: ", save_path)
end


# Example usage
spectral_linear = [1.23, 0.87, 0.68, 0.67, 0.64, 0.62, 0.61, 0.6, 0.57, 0.56]
spectral_marmousi = [1.46, 1.27, 0.79, 0.71, 0.70, 0.66, 0.64, 0.63, 0.63, 0.64]

plot_spectral_radius(
    spectral_linear,
    spectral_marmousi;
    save_path="$(@__DIR__)/spectralRadiusComparison"
)