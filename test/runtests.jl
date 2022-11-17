using FourierPoissonSolvers
using Test
using ForwardDiff

using FourierPoissonSolvers:zeromean!

include("utils.jl")

@testset "FourierPoissonSolvers.jl" begin
    include("1d.jl")
    include("2d.jl")
end
