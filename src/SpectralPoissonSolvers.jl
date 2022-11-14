module SpectralPoissonSolvers

import FFTW: r2r!, fftfreq, DHT, REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01, RODFT10, RODFT11

frequency_response(ω) = -2 + 2cos(ω)

"""
    zeromean!(arr)

Add a constant to `arr` so that the average of its entries becomes zero. This is necessary when testing solutions of
Poisson equations with only Neumann or periodic boundary conditions, since then the solution is only defined up to
addition/subtraction by a constant, so to test the accuracy of a method both the exact and approximate solutions must be
shifted first.
"""
function zeromean!(arr)
    arr .-= sum(arr) / length(arr)
end

# A struct so that creating an axis that is offset can be done by adding an `Offset()` argument, instead of a less clear `true`
struct Offset end

# Periodic boundary conditions, which require no additional data
struct Periodic end

# Dirichlet boundary conditions with the given values. Can be either nothing, a constant, or an array
struct Dirichlet
    values
end

# Neumann boundary conditions with the given values. Can be either nothing, a constant, or an array.
# Note that right now, the values are interpreted as partial derivatives, rather than normal derivatives.
# This means that compared to the normal derivative formulation, the signs are swapped one of the two boundaries.
struct Neumann
    values
end

struct Axis
    x1
    x2
    size
    pitch  # maybe call this something like 'step'
    offset # should be renamed to 'kind' or something
    bc
end

function axis(x1, x2, size, bc, offset=nothing)
    length = x2 - x1
    is_offset = offset isa Offset
    pitch = if bc isa Periodic || is_offset
        length / size
    else
        logical_size = size - 1
        if bc[1] isa Dirichlet
            logical_size += 1
        end
        if bc[2] isa Dirichlet
            logical_size += 1
        end
        length / logical_size
    end
    
    Axis(
        x1,
        x2,
        size,
        pitch,
        offset,
        bc
    )
end

function xvalues(axis::Axis)
    if axis.offset isa Offset || axis.bc isa Periodic
        range(axis.x1, axis.x2, length=2axis.size + 1)[2:2:end]
    else
        bc = axis.bc
        if bc isa Tuple{Dirichlet, Dirichlet}
            range(axis.x1, axis.x2, length=axis.size+2)[2:end-1]
        elseif bc isa Tuple{Dirichlet, Neumann}
            range(axis.x1, axis.x2, length=axis.size+1)[2:end]
        elseif bc isa Tuple{Neumann, Dirichlet}
            range(axis.x1, axis.x2, length=axis.size+1)[1:end-1]
        elseif bc isa Tuple{Neumann, Neumann}
            range(axis.x1, axis.x2, length=axis.size)
        end
    end
end

struct PoissonProblemCFG
    axes
end

struct PoissonProblem
    axes
    rhs
end

"""
    update_bcs!(rhs, i, axis)

Process all boundary conditions by adding their respective terms to the right-hand side of the linear system of equations.
"""
function update_bcs!(correction, rhs, i, axis)
    # update the array by adding terms based on the boundary conditions
    # for each axis, this possibly means adding some term to the rhs, based on the type of boundary condition and whether the grid is offset
    if axis.bc isa Periodic
        return
    end
    D = ndims(correction)
    c1 = ntuple(_ -> Colon(), i-1)
    c2 = ntuple(_ -> Colon(), D-i)
    sz = size(correction, i)
    view_left = view(correction, c1..., 1:1, c2...)
    view_right = view(correction, c1..., sz:sz, c2...)
    update_left_boundary!(view_left, axis.pitch, axis.bc[1], axis.offset)
    update_right_boundary!(view_right, axis.pitch, axis.bc[2], axis.offset)
end

function do_transform!(rhs, i, axis)
    # do the correct transform over the correct dimension
    kind = fwd_transform(axis.bc, axis.offset)
    r2r!(rhs, kind, i)
end

function scale_coefficients!(rhs, axes, is_singular)
    # scale the array, which is now coefficients of the transform, by the inverse eigenvalue and scaling factor of the transform
    scale = zeros(eltype(rhs), size(rhs))
    for (i, axis) in enumerate(axes)
        n = size(rhs, i)
        ls = frequency_response.(frequencies(axis.bc, axis.offset, n)) ./ axis.pitch^2
        scale .+= reshape(ls, ntuple(_ -> 1, i-1)..., :)
    end
    rhs ./= scale
    for (i, axis) in enumerate(axes)
        n = size(rhs, i)
        rhs ./= scalingfactor(axis.bc, axis.offset, n)
    end
    if is_singular
        rhs[1] = 0.0
    end
end

function do_inverse_transform!(rhs, i, axis)
    # transform the coefficients back into a spatial thing
    kind = bwd_transform(axis.bc, axis.offset)
    r2r!(rhs, kind, i)
end

function solve(prob)
    rhs = Array(prob.rhs)
    correction = zeros(eltype(rhs), size(rhs))
    for (i, axis) in enumerate(prob.axes)
        update_bcs!(correction, rhs, i, axis)
    end
    rhs .+= correction
    for (i, axis) in enumerate(prob.axes)
        do_transform!(rhs, i, axis)
    end
    scale_coefficients!(rhs, prob.axes, is_singular(prob))
    for (i, axis) in enumerate(prob.axes)
        do_inverse_transform!(rhs, i, axis)
    end
    if is_singular(prob)
        # If the problem is singular, then setting the coefficient corresponding to the constant term does not ensure
        # that the resulting `rhs` has zero average value, since for the RODFT00 transform (i.e. double Neumann bc's on
        # a normal grid) some of the higher frequencies do not sum to zero. So to ensure that the result sums to zero,
        # we shift the solution manually after doing the inverse transform.
        zeromean!(rhs)
    end
    rhs
end

function is_singular(prob)
    for axis in prob.axes
        bc = axis.bc
        if bc isa Periodic
            continue
        end
        if bc[1] isa Dirichlet || bc[2] isa Dirichlet
            return false
        end
    end
    true
end

"""
    exact_for_quadratic_solutions(prob)

Returns true if the solution to this problem is correct up to numerical precision (instead of just 2nd-order accurate)
when the right hand side is constant, meaning that the solution is given by a polynomial of degree 2 or lower. In
general, second-order accurate methods should also be able to solve such problems exactly, but this is not always the
case here due to a technicality. Specifically, if any of the axes use an offset grid and Dirichlet boundary conditions
on one or both sides, then the solution will still be second-order accurate but not exact for solutions that are given
by a quadratic function. This is due to the fact that in such cases, the correct discretisation does not exactly line up
with a Discrete Sine Transform (for more info, see the block comment in `dirichlet.jl`).
"""
function exact_for_quadratic_solutions(prob)
    for axis in prob.axes
        bc = axis.bc
        if bc isa Periodic || axis.offset isa Nothing
            continue
        elseif bc[1] isa Dirichlet || bc[2] isa Dirichlet
            return false
        end
    end
    true
end

include("periodic.jl")
include("dirichlet.jl")
include("neumann.jl")
include("mixed.jl")

export Periodic, Dirichlet, Neumann, Offset, PoissonProblem, solve, axis, is_singular, exact_for_quadratic_solutions

end
