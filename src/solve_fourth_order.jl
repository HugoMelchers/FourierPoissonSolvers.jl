function add_correction!(b, prob::PoissonProblem{T,N,Val{4}}, x) where {T,N}
    D = length(prob.size)
    for dim in 1:D
        sz = prob.size[dim]
        (bc_l, bc_r) = prob.boundaries[dim]
        b1_left = selectdim(b, dim, 1)
        b2_left = selectdim(b, dim, 2)
        b1_right = selectdim(b, dim, sz)
        b2_right = selectdim(b, dim, sz - 1)
        x_rev = selectdim(x, dim, sz:-1:1)
        h = prob.step[dim]
        g = prob.grid[dim]
        correction!(b1_left, b2_left, x, dim, h, bc_l, g)
        correction!(b1_right, b2_right, x_rev, dim, h, bc_r, g)
    end
end

function solve_in_place_periodic!(arr, prob::PoissonProblem)
    mul!(arr, prob.fwd_plan, arr)
    arr .*= prob.coefficients
    mul!(arr, prob.bwd_plan, arr)
    arr
end

function solve(prob::PoissonProblem{T,N,Val{4}}, arr::Array) where {T,N}
    # If the problem has periodic boundary conditions everywhere, then it doesn't make sense to use the iterative
    # fourth-order method since we can obtain spectral accuracy with a direct method
    if fully_periodic(prob)
        return solve_in_place_periodic!(copy(arr), prob)
    end
    # Else, iterate `x = Ã₄ \ (b - D*x)` where `Ã₄` is the FFT-based mostly-fourth-order scheme, and D is the correction
    x = zeros(eltype(arr), size(arr))
    b2 = similar(arr)
    for r in prob.relaxationparameters
        b2 .= arr
        add_correction!(b2, prob, x)
        mul!(b2, prob.fwd_plan, b2)
        b2 .*= prob.coefficients
        mul!(b2, prob.bwd_plan, b2)
        @. x = (1 - r) * x + r * b2
    end
    is_singular(prob) && zeromean!(x)
    x
end

function solve(prob::PoissonProblem{T,N,Val{4}}, a::WithBoundaries) where {T,N}
    a1 = copy(a.rhs)
    D = length(prob.size)
    for i in 1:D
        sz = prob.size[i]
        (v_left, v_right) = a.bvs[i]
        (c_left, c_right) = prob.boundaries[i]
        if c_left isa Periodic
            continue
        end
        h = prob.step[i]
        g = prob.grid[i]
        a1_left = selectdim(a1, i, 1)
        a2_left = selectdim(a1, i, 2)
        a1_right = selectdim(a1, i, sz)
        a2_right = selectdim(a1, i, sz - 1)
        v_left !== nothing && add_boundary_term_4th_order!(a1_left, a2_left, c_left, g, v_left, h)
        v_right !== nothing && add_boundary_term_4th_order!(a1_right, a2_right, c_right, g, v_right, -h)
    end
    solve(prob, a1)
end

function correction!(b1, b2, a, dim, h, bc, grid)
    (coeffs1, coeffs2) = correction_coefficients(bc, grid)
    if coeffs1 !== nothing
        for i in 1:5
            c = coeffs1[i] / h^2
            b1 .-= c .* selectdim(a, dim, i)
        end
    end
    if coeffs2 !== nothing
        for i in 1:5
            c = coeffs2[i] / h^2
            b2 .-= c .* selectdim(a, dim, i)
        end
    end
end
