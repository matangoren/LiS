using Helmholtz
using PyPlot
using Printf
using KrylovMethods
using LinearAlgebra
using Flux
using DelimitedFiles

include("../auxiliary.jl")
current_file_dir = dirname(@__FILE__)

##########################################################################
# Linear model plot ######################################################
##########################################################################

n = [128, 128]*2 .+ 1;
n = [128,256]
m_coeff = [0.25, 1.0]

lower = upper = m_coeff[1]
if length(m_coeff) == 2
    upper = m_coeff[2]
end

m = linear_grid_ratio(lower, upper, n)

# Create a gridspec layout: 1 row, 2 columns
fig = figure(figsize=(4, 4))
gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)

# Main image axis
ax = fig.add_subplot(gs[1])
img = ax.imshow(m, cmap="viridis")
ax.set_xticks([])
ax.set_yticks([])

# Colorbar axis
cax = fig.add_subplot(gs[2])
colorbar(img, cax=cax)
savefig("plots/linear_model.eps")
savefig("plots/linear_model.png")



