using LinearAlgebra
using PyPlot


LapSymbol(theta,h) = (2/h^2)*(2-cos(theta[1])-cos(theta[2]));
Mass(omega,kappa,gamma) = - omega^2 * kappa^2 * (1 - gamma * im); 

HSymbol(theta,h,omega,kappa,gamma) = LapSymbol(theta,h) + Mass(omega,kappa,gamma);
HslSymbol(theta,alpha,h,omega,kappa,gamma) = HSymbol(theta,h,omega,kappa,gamma) + alpha * omega^2 * kappa^2 * im;


function SpecRad(gamma,kappa,kappa0,alpha,h)
    # calculating the spectral radius of the error propagation matrix for the multipreconditioning step

    ht = 0.1
    theta1 = collect(-pi/2:ht:pi/2)
    theta2 = theta1

    spec_rad_arr = zeros(length(theta1), length(theta2))
    spec_rad_arr_sl = zeros(length(theta1), length(theta2))
    spec_rad_arr_lis = zeros(length(theta1), length(theta2)) 

    for i = 1:length(theta1)
        for j = 1:length(theta2)
            theta = [theta1[i] theta2[j]]

            # SL_propagation_symbol = (1 - HSymbol(theta,h,omega,kappa,gamma)/HslSymbol(theta,alpha,h,omega,kappa,gamma));
            # LiS_propagation_symbol = (1 - HSymbol(theta,h,omega,kappa,gamma)/HSymbol(theta,h,omega,kappa0,gamma0));
            LiS_propagation_symbol = (1 - HSymbol(theta,h,omega,kappa,gamma)/HslSymbol(theta,alpha,h,omega,kappa0,gamma));
            # spec_rad_arr[i,j] = abs(SL_propagation_symbol * LiS_propagation_symbol);
            # spec_rad_arr_sl[i,j] = abs(SL_propagation_symbol);
            spec_rad_arr_lis[i,j] = abs(LiS_propagation_symbol);
            # spec_rad_arr[i,j] = abs(LiSL_propagation_symbol);
        end
    end

    n = length(theta1)
    spec_rad = maximum(spec_rad_arr)

    return spec_rad, spec_rad_arr, spec_rad_arr_sl, spec_rad_arr_lis, theta1, theta2
end

PyPlot.close("all")

################## Parameters: ######################
h = 1 / 128;
kappa0 = 1; 
gamma0 = 0.5;
gppw = 10;
omega = (2 * pi)/(h * gppw); # frequency
gamma = 0.01*pi; # attenuation
alpha = 0.5; # added shift
kappa = 0.9; 


####################################### Experiments #######################################

################# contour #####################

spec_rad, spec_rad_arr, spec_rad_arr_sl, spec_rad_arr_lis, theta1, theta2 = SpecRad(gamma,kappa,kappa0,alpha,h)

# fig, ax = PyPlot.subplots()
# cont = ax.contour(theta1, theta2, spec_rad_arr, 20);
# fig.colorbar(cont);
# # m = length(cont.levels);
# wanted_levels = cont.levels[1:2:end];
# ax.clabel(cont, wanted_levels)
# # ax.clabel(cont, cont.levels)
# ax.set_title("Symbol of the multipreconditioning error propagation matrix")
# ax.set_xlabel(L"\theta_1")
# ax.set_ylabel(L"\theta_2")
# ax.set_xticks([-pi / 2, 0, pi / 2], [L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}"])
# ax.set_yticks([-pi / 2, 0, pi / 2], [L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}"])
# fig.show()

# fig, ax = PyPlot.subplots()
# cont = ax.contour(theta1, theta2, spec_rad_arr_sl, 20);
# fig.colorbar(cont);
# # m = length(cont.levels);
# wanted_levels = cont.levels[1:2:end];
# ax.clabel(cont, wanted_levels)
# # ax.clabel(cont, cont.levels)
# ax.set_title("Symbol of the multipreconditioning error propagation matrix, SL")
# ax.set_xlabel(L"\theta_1")
# ax.set_ylabel(L"\theta_2")
# ax.set_xticks([-pi / 2, 0, pi / 2], [L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}"])
# ax.set_yticks([-pi / 2, 0, pi / 2], [L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}"])
# fig.show()

fig, ax = PyPlot.subplots()
cont = ax.contour(theta1, theta2, spec_rad_arr_lis, 20);
fig.colorbar(cont);
# m = length(cont.levels);
wanted_levels = cont.levels[1:2:end];
ax.clabel(cont, wanted_levels)
# ax.clabel(cont, cont.levels)
ax.set_title("Symbol of the multipreconditioning error propagation matrix, LiS")
ax.set_xlabel(L"\theta_1")
ax.set_ylabel(L"\theta_2")
ax.set_xticks([-pi / 2, 0, pi / 2], [L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}"])
ax.set_yticks([-pi / 2, 0, pi / 2], [L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}"])
fig.show()


########################### "integration" ###########################

# kappa_arr = 0.2:0.1:1;
# n = length(kappa_arr);
# spec_arr = zeros(n);
# for i = 1:n
#     _, spec_arr[i], _, _ = SpecRad(gamma,kappa_arr[i],kappa0,alpha,h)
# end

# #fig, ax1 = PyPlot.subplots(figsize=(2.6,2))
# fig, ax1 = PyPlot.subplots()
# ax1.plot(kappa_arr, apec_arr, label=L"label");
# ax1.legend()
# ax1.set_title("spectral radius vs. kappa");
# ax1.set_xlabel(L"\kappa");
# ax1.set_ylabel(L"\rho");
# fig.show()