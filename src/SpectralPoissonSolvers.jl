module SpectralPoissonSolvers

import FFTW: r2r!, fftfreq, DHT, REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01, RODFT10, RODFT11

# A struct so that creating an axis that is offset can be done by adding an `Offset()` argument, instead of a less clear `true`
struct Offset end

# Periodic boundary conditions, which require no additional data
struct Periodic end

abstract type AbstractBoundaryCondition end

# Dirichlet boundary conditions with the given values. Can be either nothing, a constant, or an array
struct Dirichlet{V}
    values::V
end

# Neumann boundary conditions with the given values. Can be either nothing, a constant, or an array.
# Note that right now, the values are interpreted as partial derivatives, rather than normal derivatives.
# This means that compared to the normal derivative formulation, the signs are swapped for the right boundary.
struct Neumann{V}
    values::V
end

struct Axis
    length
    size
    pitch
    is_offset
    bc
end

function axis(length, size, bc, offset=nothing)
    is_offset = offset isa Offset
    pitch = if bc isa Periodic || is_offset
        length / size
    else
        logical_size = size + 1
        if bc[1] isa Dirichlet
            logical_size -= 1
        end
        if bc[2] isa Dirichlet
            logical_size -= 1
        end
        length / logical_size
    end
    
    Axis(
        length,
        size,
        pitch,
        is_offset,
        bc
    )
end

struct PoissonProblemCFG
    axes
end

struct PoissonProblem
    axes
    rhs
end

function update_bc_left!(rhs, i, bc, pitch, is_offset)
    D = ndims(rhs)
    c1 = ntuple(_ -> Colon(), i-1)
    c2 = ntuple(_ -> Colon(), D-i)
    if bc isa Dirichlet
        if bc.values !== nothing
            mult = ifelse(is_offset, 2.0, 1.0)
            rhs[c1..., 1:1, c2...] .-= bc.values .* (mult / pitch^2)
        end
    else # bc isa Neumann
        if bc.values !== nothing
            mult = ifelse(is_offset, 1.0, 2.0)
            rhs[c1..., 1:1, c2...] .+= bc.values .* (mult / pitch)
        end
    end
end

function update_bc_right!(rhs, i, bc, pitch, is_offset)
    D = ndims(rhs)
    c1 = ntuple(_ -> Colon(), i-1)
    c2 = ntuple(_ -> Colon(), D-i)
    if bc isa Dirichlet
        if bc.values !== nothing
            mult = ifelse(is_offset, 2.0, 1.0)
            rhs[c1..., end:end, c2...] .-= bc.values .* (mult / pitch^2)
        end
    else # bc isa Neumann
        if bc.values !== nothing
            mult = ifelse(is_offset, 1.0, 2.0)
            rhs[c1..., end:end, c2...] .-= bc.values .* (mult / pitch)
        end
    end
end

function update_bcs!(rhs, i, axis)
    # update the array by adding terms based on the boundary conditions
    # for each axis, this possibly means adding some term to the rhs, based on the type of boundary condition and whether the grid is offset
    if axis.bc isa Periodic
        return
    end
    update_bc_left!(rhs, i, axis.bc[1], axis.pitch, axis.is_offset)
    update_bc_right!(rhs, i, axis.bc[2], axis.pitch, axis.is_offset)
end

function do_transform!(rhs, i, axis)
    # do the correct transform over the correct dimension
    kind = which_fft(axis.bc, axis.is_offset)
    r2r!(rhs, kind, i)
end

function scale_coefficients!(rhs, axes, is_singular)
    # scale the array, which is now coefficients of the transform, by the inverse eigenvalue and scaling factor of the transform
    scale = zeros(eltype(rhs), size(rhs))
    for (i, axis) in enumerate(axes)
        n = size(rhs, i)
        t = which_fft(axis.bc, axis.is_offset)
        ls = evs(t, n) ./ axis.pitch^2
        scale .+= reshape(ls, ntuple(_ -> 1, i-1)..., :)
    end
    rhs ./= scale
    for (i, axis) in enumerate(axes)
        t = which_fft(axis.bc, axis.is_offset)
        n = size(rhs, i)
        rhs ./= scaling_factor(t, n)
    end
    if is_singular
        rhs[1] = 0.0
    end
end

function do_inverse_transform!(rhs, i, axis)
    # transform the coefficients back into a spatial thing
    fwd_kind = which_fft(axis.bc, axis.is_offset)
    kind = inverse_transform_of(fwd_kind)
    r2r!(rhs, kind, i)
end

function solve(prob)
    rhs = Array(prob.rhs)
    for (i, axis) in enumerate(prob.axes)
        update_bcs!(rhs, i, axis)
    end
    for (i, axis) in enumerate(prob.axes)
        do_transform!(rhs, i, axis)
    end
    scale_coefficients!(rhs, prob.axes, is_singular(prob))
    for (i, axis) in enumerate(prob.axes)
        do_inverse_transform!(rhs, i, axis)
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

include("transforms.jl")

export Periodic, Dirichlet, Neumann, Offset, PoissonProblem, solve, axis, is_singular

end
