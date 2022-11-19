module FourierPoissonSolvers

import FFTW: r2rFFTWPlan, plan_r2r!, fftfreq, DHT, REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01, RODFT10, RODFT11
import Base:\
import LinearAlgebra:mul!

# A struct so that creating an axis that is offset can be done by adding an `Offset()` argument, instead of a less clear `true`
struct Offset end

frequency_response(ω) = -2 + 2cos(ω)

include("periodic.jl")
include("dirichlet.jl")
include("neumann.jl")
include("mixed.jl")
include("rhs.jl")
include("poissonproblem.jl")
include("api.jl")

export Periodic, Dirichlet, Neumann, Offset, PoissonProblem, solve, is_singular, exact_for_quadratic_solutions, with_boundaries, nodes

end
