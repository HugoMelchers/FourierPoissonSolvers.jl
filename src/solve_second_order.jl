function add_boundary_terms!(f, i, step, bc, values, grid)
    #=
    The Poisson equation `Δu=f` with boundary conditions is discretised into a linear system `Au=f+Δf` where `Δf` is an
    additional term that corrects for the boundary conditions. This function adds that correction to `f`.
    =#
    bc[1] isa Periodic && return
    sz = size(f, i)
    view_left = selectdim(f, i, 1)
    view_right = selectdim(f, i, sz)
    values[1] !== nothing && add_boundary_term!(view_left, bc[1], grid, values[1], step)
    values[2] !== nothing && add_boundary_term!(view_right, bc[2], grid, values[2], -step)
end

"""
    solve(prob::PoissonProblem{T,N,Val{2}}, arr::Array) where {T,N}

Solves the `PoissonProblem` `prob` (containing boundary conditions, grid sizes, etc.) with the right-hand side `rhs` and
homogeneous boundary conditions. The dimensions and element type of `arr` must match those of `prob`.
"""
function solve(prob::PoissonProblem{T,N,Val{2}}, arr::Array) where {T,N}
    solve_in_place!(copy(arr), prob)
end

"""
    solve_in_place!(arr, prob::PoissonProblem{T,N,Val{2}}) where {T,N}

Solve the second-order Poisson equation in-place, i.e. by overwriting `arr`.
"""
function solve_in_place!(arr, prob::PoissonProblem{T,N,Val{2}}) where {T,N}
    mul!(arr, prob.fwd_plan, arr)
    arr .*= prob.coefficients
    mul!(arr, prob.bwd_plan, arr)
    # If the problem is singular, then setting the coefficient corresponding to the constant term does not ensure
    # that the resulting `rhs` has zero average value, since for the RODFT00 transform (i.e. double Neumann bc's on
    # a normal grid) some of the higher frequencies do not sum to zero. So to ensure that the result sums to zero,
    # we shift the solution manually after doing the inverse transform.
    is_singular(prob) && zeromean!(arr)
    arr
end

"""
    solve(prob::PoissonProblem{T,N,Val{2}}, right_hand_side::WithBoundaries) where {T,N}

Solve the `PoissonProblem` `prob` with the given right-hand side, containing the right-hand side of the PDE as well as
boundary values.
"""
function solve(prob::PoissonProblem{T,N,Val{2}}, right_hand_side::WithBoundaries) where {T,N}
    arr_copy = copy(right_hand_side.rhs)
    for i in 1:length(prob.size)
        add_boundary_terms!(arr_copy, i, prob.step[i], prob.boundaries[i], right_hand_side.bvs[i], prob.grid[i])
    end
    solve_in_place!(arr_copy, prob)
end

"""
    \\(prob::PoissonProblem, rhs)

Equivalent to `solve(prob, rhs)`.
"""
\(prob::PoissonProblem, rhs) = solve(prob, rhs)
