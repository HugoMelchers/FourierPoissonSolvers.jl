"""
    solve(prob::PoissonProblem, rhs)

Solves the `PoissonProblem` `prob` (containing boundary conditions, grid sizes, etc.) with the right-hand side `rhs`.
The dimensions of `rhs` must match `prob.size`.
"""
function solve(prob::PoissonProblem, rhs)
    rhs = copy(rhs)
    for i in 1:length(prob.size)
        update_bcs!(rhs, i, prob.step[i], prob.boundaries[i], prob.grid[i])
    end
    mul!(rhs, prob.fwd_plan, rhs)
    scale_coefficients!(rhs, prob)
    mul!(rhs, prob.bwd_plan, rhs)
    if is_singular(prob)
        # If the problem is singular, then setting the coefficient corresponding to the constant term does not ensure
        # that the resulting `rhs` has zero average value, since for the RODFT00 transform (i.e. double Neumann bc's on
        # a normal grid) some of the higher frequencies do not sum to zero. So to ensure that the result sums to zero,
        # we shift the solution manually after doing the inverse transform.
        zeromean!(rhs)
    end
    rhs
end

"""
    \\(prob::PoissonProblem, rhs)

Equivalent to `solve(prob, rhs)`.
"""
\(prob::PoissonProblem, rhs) = solve(prob, rhs)

"""
    is_singular(prob::PoissonProblem)

Returns true if the given problem is singular, meaning that it has periodic or Neumann boundary conditions in all
axes. In this case, there no Dirichlet boundary condition to fix the value of the solution at any point, meaning that
the solution is only defined up to addition or subtraction by a constant. Furthermore, a solution may not exist if
the right-hand side and boundary conditions don't satisfy a specific equation (known as the compatibility relation).
For example, in the one-dimensional case `u'' = 1'` for `0 < x < 1`, it holds that `u'(1) = u'(0) + 1`. For periodic
boundary conditions, this does not have a solution since the periodicity requires `u'(0) = u'(1)`. For Neumann boundary
conditions  `u'(0) = a` and `u'(1) = b`, there is only a solution if `b = a + 1` .

If the problem is singular, the solution obtained is equivalent to using the Moore-Penrose pseudo-inverse, which means:

- in the equation `Δu = f`, first `f` is shifted by a constant to give the problem a solution
- the resulting problem has infinitely many solutions, and the returned solution is the one whose average value over the domain is zero
"""
function is_singular(prob::PoissonProblem)
    for bc in prob.boundaries
        if bc.left isa Dirichlet || bc.right isa Dirichlet
            return false
        end
    end
    true
end

"""
    exact_for_quadratic_solutions(prob::PoissonProblem)

Returns true if the solution to this problem is correct up to numerical precision (instead of just 2nd-order accurate)
when the right hand side is constant, meaning that the solution is given by a polynomial of degree 2 or lower. In
general, second-order accurate methods should also be able to solve such problems exactly, but this is not always the
case here due to a technicality. Specifically, if any of the axes use an offset grid and Dirichlet boundary conditions
on one or both sides, then the solution will still be second-order accurate but not exact for solutions that are given
by a quadratic function. This is due to the fact that in such cases, the correct discretisation does not exactly line up
with a Discrete Sine Transform (for more info, see the block comment in `dirichlet.jl`).
"""
function exact_for_quadratic_solutions(prob::PoissonProblem)
    for (bc, grid) in zip(prob.boundaries, prob.grid)
        if bc isa Periodic || grid isa Nothing
            continue
        elseif bc.left isa Dirichlet || bc.right isa Dirichlet
            return false
        end
    end
    true
end

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
