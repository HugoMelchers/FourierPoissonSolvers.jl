using SpectralPoissonSolvers
using Test
using ForwardDiff

using SpectralPoissonSolvers:zeromean!, xvalues

include("utils.jl")

@testset "SpectralPoissonSolvers.jl" begin
    include("1d.jl")
    include("2d.jl")
end
