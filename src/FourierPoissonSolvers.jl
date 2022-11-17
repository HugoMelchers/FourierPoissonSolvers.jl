module FourierPoissonSolvers

import FFTW: r2rFFTWPlan, plan_r2r!, fftfreq, DHT, REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01, RODFT10, RODFT11
import Base:\
import LinearAlgebra:mul!

frequency_response(ω) = -2 + 2cos(ω)

include("poissonproblem.jl")

function update_bcs!(rhs, i, step, bc, values, grid)
    # update the array by adding terms based on the boundary conditions
    # for each axis, this possibly means adding some term to the rhs, based on the type of boundary condition and whether the grid is offset
    bc[1] isa Periodic && return
    D = ndims(rhs)
    c1 = ntuple(_ -> Colon(), i-1)
    c2 = ntuple(_ -> Colon(), D-i)
    sz = size(rhs, i)
    view_left = view(rhs, c1..., 1:1, c2...)
    view_right = view(rhs, c1..., sz:sz, c2...)
    values[1] !== nothing && update_left_boundary!(view_left, step, bc[1], values[1], grid)
    values[2] !== nothing && update_right_boundary!(view_right, step, bc[2], values[2], grid)
end

include("rhs.jl")
include("api.jl")
include("periodic.jl")
include("dirichlet.jl")
include("neumann.jl")
include("mixed.jl")

export Periodic, Dirichlet, Neumann, Offset, PoissonProblem, solve, is_singular, exact_for_quadratic_solutions, with_boundaries

end
