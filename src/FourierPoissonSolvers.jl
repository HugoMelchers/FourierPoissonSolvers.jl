module FourierPoissonSolvers

import FFTW
import FFTW: r2rFFTWPlan, plan_r2r!, fftfreq, DHT, REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01, RODFT10, RODFT11
const FFTW_ESTIMATE = FFTW.ESTIMATE
const FFTW_MEASURE = FFTW.MEASURE
const FFTW_PATIENT = FFTW.PATIENT
const FFTW_EXHAUSTIVE = FFTW.EXHAUSTIVE
import Base: \
import LinearAlgebra: mul!

# A struct so that creating an axis that is offset can be done by adding an `Offset()` argument, instead of a less clear `true`
struct Offset end

include("boundaries.jl")
include("poissonproblem.jl")
include("right_hand_side.jl")
include("solve_second_order.jl")
include("solve_fourth_order.jl")

export Periodic, Dirichlet, Neumann, Offset, PoissonProblem, solve, is_singular, exact_for_quadratic_solutions, with_boundaries, nodes, FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT
end
