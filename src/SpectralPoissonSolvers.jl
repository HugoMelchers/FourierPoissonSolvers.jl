module SpectralPoissonSolvers

import FFTW: r2r!, plan_r2r!, fftfreq, DHT, REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01, RODFT10, RODFT11
import Base:\
import LinearAlgebra:mul!

frequency_response(ω) = -2 + 2cos(ω)

# A struct so that creating an axis that is offset can be done by adding an `Offset()` argument, instead of a less clear `true`
struct Offset end

# Periodic boundary conditions, which require no additional data
struct Periodic end

# Dirichlet boundary conditions with the given values. Can be either nothing, a constant, or an array
mutable struct Dirichlet
    values
end

# Neumann boundary conditions with the given values. Can be either nothing, a constant, or an array.
# Note that right now, the values are interpreted as partial derivatives, rather than normal derivatives.
# This means that compared to the normal derivative formulation, the signs are swapped one of the two boundaries.
mutable struct Neumann
    values
end

struct Boundary
    left
    right
end

node_range(_, _, ::Offset, n) = (2n+1, 2:2:(2n))
node_range(::Periodic, ::Periodic, ::Nothing, n) = (2n+1, 2:2:(2n))
node_range(left::Union{Dirichlet, Neumann}, right::Union{Dirichlet, Neumann}, ::Nothing, n) = begin
    logical_size = n
    skip_left = 0
    skip_right = 0
    if left isa Dirichlet
        # Dirichlet on left boundary: exclude that point
        logical_size += 1
        skip_left = 1
    end
    if right isa Dirichlet
        logical_size += 1
        skip_right = 1
    end
    range = (1 + skip_left):(logical_size - skip_right)
    (logical_size, range)
end
function nodes(left, right, grid, n, x1, x2)
    logical_size, _range = node_range(left, right, grid, n)
    range(x1, x2, length=logical_size)[_range]
end
function step(left, right, grid, n, x1, x2)
    (logical_size, range) = node_range(left, right, grid, n)
    stride = if range isa UnitRange
        1
    else
        range.step
    end
    stride * (x2 - x1) / (logical_size - 1)
end

struct PoissonProblem
    size
    step
    boundaries
    grid
    lims
    nodes
    coefficients
    fwd_plan
    bwd_plan
end

"""
    PoissonProblem(size; boundaries, lims, grid)
"""
function PoissonProblem(size; boundaries, lims, grid)
    _nodes = ntuple(
        i -> nodes(boundaries[i].left, boundaries[i].right, grid[i], size[i], lims[i][1], lims[i][2]),
        length(size)
    )
    _step = ntuple(
        i -> step(boundaries[i].left, boundaries[i].right, grid[i], size[i], lims[i][1], lims[i][2]),
        length(size)
    )
    coefficients = ntuple(
        i -> frequency_response.(frequencies(boundaries[i].left, boundaries[i].right, grid[i], size[i])) ./ _step[i]^2,
        length(size)
    )
    
    fwd_plan = plan_fwd(size, boundaries, grid)
    bwd_plan = plan_bwd(size, boundaries, grid)
    
    PoissonProblem(
        size, _step, boundaries, grid, lims, _nodes, coefficients, fwd_plan, bwd_plan
    )
end

function plan_fwd(size, boundaries, grid)
    kinds = ntuple(
        i -> fwd_transform(boundaries[i].left, boundaries[i].right, grid[i]),
        length(size)
    )
    plan_r2r!(zeros(size...), kinds)
end

function plan_bwd(size, boundaries, grid)
    kinds = ntuple(
        i -> bwd_transform(boundaries[i].left, boundaries[i].right, grid[i]),
        length(size)
    )
    plan_r2r!(zeros(size...), kinds)
end

function update_bcs!(rhs, i, step, bc, grid)
    # update the array by adding terms based on the boundary conditions
    # for each axis, this possibly means adding some term to the rhs, based on the type of boundary condition and whether the grid is offset
    if bc.left isa Periodic
        return
    end
    D = ndims(rhs)
    c1 = ntuple(_ -> Colon(), i-1)
    c2 = ntuple(_ -> Colon(), D-i)
    sz = size(rhs, i)
    view_left = view(rhs, c1..., 1:1, c2...)
    view_right = view(rhs, c1..., sz:sz, c2...)
    bc.left.values !== nothing && update_left_boundary!(view_left, step, bc.left, grid)
    bc.right.values !== nothing && update_right_boundary!(view_right, step, bc.right, grid)
end

function scale_coefficients!(rhs, prob::PoissonProblem)
    # scale the array, which is now coefficients of the transform, by the inverse eigenvalue and scaling factor of the transform
    scale = zeros(eltype(rhs), size(rhs))
    for i in 1:length(prob.size)
        ls = prob.coefficients[i]
        scale .+= reshape(ls, ntuple(_ -> 1, i-1)..., :)
    end
    rhs ./= scale
    for i in 1:length(prob.size)
        n = size(rhs, i)
        rhs ./= scalingfactor(prob.boundaries[i].left, prob.boundaries[i].right, prob.grid[i], n)
    end
    if is_singular(prob)
        rhs[1] = 0.0
    end
end

# allow solving as `u = prob \ f`
include("api.jl")
include("periodic.jl")
include("dirichlet.jl")
include("neumann.jl")
include("mixed.jl")

export Periodic, Dirichlet, Neumann, Offset, PoissonProblem, solve, is_singular, exact_for_quadratic_solutions, Boundary

end
