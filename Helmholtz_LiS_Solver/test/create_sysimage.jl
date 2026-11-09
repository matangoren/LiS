using PackageCompiler

create_sysimage(
    [
        :Helmholtz,
        :KrylovMethods,
        :Multigrid,
        :Flux,
        :PyPlot,
        Symbol("jInv")
    ];
    sysimage_path="LiS_sysimage.so",
    precompile_execution_file="./test/warmup.jl"
)