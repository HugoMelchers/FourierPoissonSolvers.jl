#=
Attempt #2 at solving the fourth-order Poisson equation. Instead of iterating `x <- x - A2 \ (A4*x - b)`, I iterate
`x <- x - Ã₄ \ (A₄*x - b)`, where `Ã₄` is the matrix that is fourth-order accurate for the Poisson equation, with
exception of the boundary conditions. Since `Ã₄` and `A₄` are the same except at the boundaries, this iteration can also
be written as `x <- Ã₄ \ (b - dA₄*x)`, where `dA₄ = A₄ - Ã₄`. Then, the fixed-point satisfies `x = Ã₄ \ (b - dA₄*x)`,
which can be rewritten into `Ã₄*x = b - dA₄*x` so `(Ã₄ + dA₄)*x = b` , and since `Ã₄ + dA₄ = A₄`, this fixed point is
the solution to the fourth-order accurate discretisation.
=#

struct FourthOrder{T,N,BCS,G,Plan1,Plan2}
    size::NTuple{N,Int}
    step::NTuple{N,T}
    boundaries::BCS
    grid::G
    lims::NTuple{N,NTuple{2,T}}
    coefficients::Array{T,N}
    fwd_plan::r2rFFTWPlan{T,Plan1,true,N,UnitRange{Int}}
    bwd_plan::r2rFFTWPlan{T,Plan2,true,N,UnitRange{Int}}
end

function FourthOrder(size::NTuple{N,Int}; boundaries, lims, grid=nothing) where {N}
    T = Float64 # TODO determine from input
    _grid = promote_grid(Val(N), grid)
    _boundaries = promote_boundary_conditions(Val(N), boundaries)
    _lims = promote_lims(T, Val(N), lims)
    left_boundaries = getindex.(_boundaries, 1)
    right_boundaries = getindex.(_boundaries, 2)
    left_lims = getindex.(_lims, 1)
    right_lims = getindex.(_lims, 2)
    _step = step.(left_boundaries, right_boundaries, _grid, size, left_lims, right_lims)
    coefficients = eigenvalues4.(left_boundaries, right_boundaries, _grid, size) ./ _step .^ 2
    _coefficients = zeros(T, size...)
    transforms_fwd = fwd_transform.(left_boundaries, right_boundaries, _grid)
    transforms_bwd = bwd_transform.(left_boundaries, right_boundaries, _grid)
    fwd_plan = plan_r2r!(_coefficients, transforms_fwd)
    bwd_plan = plan_r2r!(_coefficients, transforms_bwd)
    for i in 1:length(size)
        _coefficients .+= reshape(coefficients[i], ntuple(_ -> 1, i - 1)..., :)
    end
    _coefficients .*= prod(scalingfactor.(left_boundaries, right_boundaries, _grid, size))
    _coefficients .= 1 ./ _coefficients
    prob = FourthOrder(size, _step, _boundaries, _grid, _lims, _coefficients, fwd_plan, bwd_plan)
    is_singular(prob) && (prob.coefficients[1] = 0.0)
    prob
end

function add_correction!(b, prob::FourthOrder, x)
    D = length(prob.size)
    for i in 1:D
        c1 = ntuple(_ -> Colon(), i-1)
        c2 = ntuple(_ -> Colon(), D-i)
        sz = prob.size[i]
        (bc_l, bc_r) = prob.boundaries[i]
        b_rev = view(b, c1..., sz:-1:1, c2...)
        x_rev = view(x, c1..., sz:-1:1, c2...)
        h = prob.step[i]
        g = prob.grid[i]
        mul_boundary!(b, x, c1, c2, h, bc_l, g, false)
        mul_boundary!(b_rev, x_rev, c1, c2, h, bc_r, g, true)
    end
end

function spectral_interval(prob::FourthOrder)
    r1 = 0.0
    r2 = 0.0
    radius(::Dirichlet, ::Nothing) =  0.2577122354265
    radius(::Dirichlet, ::Offset)  = -0.38005393284828
    radius(::Neumann, ::Nothing)   = -0.857024488446
    radius(::Neumann, ::Offset)    =  0.09013067273
    for i in 1:length(prob.size)
        (left, right) = prob.boundaries[i]
        grid = prob.grid[i]
        r_left = radius(left, grid)
        r_right = radius(right, grid)
        r1 = min(r1, r_left, r_right)
        r2 = max(r2, r_left, r_right)
    end
    r1, r2
end

function optimal_relaxation_parameters(a, b, k)
    rs = [inv((a+b)/2 + (b-a)/2 * cos(pi*(j-1/2)/k)) for j in 1:k]
    total_factor = prod(1 .- rs.*a)
    rs, total_factor
end

function steps_to_error(a, b, r)
    k = 0
    err = 1
    while err > r
        k += 1
        _, err = optimal_relaxation_parameters(a, b, k)
    end
    k + 3
end

function relaxation_parameters(prob)
    r1, r2 = spectral_interval(prob)
    a, b = 1 - r2, 1 - r1
    k = steps_to_error(a, b, eps())
    optimal_relaxation_parameters(a, b, k)[1]
end

function solve(prob::FourthOrder, arr::Array)
    # iterate `x = Ã₄ \ (b - D*x)` where `Ã₄` is the FFT-based mostly-fourth-order scheme, and D is the correction
    x = zeros(eltype(arr), size(arr))
    rs = relaxation_parameters(prob)
    b2 = similar(arr)
    itr = 0
    for r in rs
        itr += 1
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

function solve(prob::FourthOrder, a::WithBoundaries)
    a1 = copy(a.rhs)
    D = length(prob.size)
    for i in 1:D
        c1 = ntuple(_ -> Colon(), i-1)
        c2 = ntuple(_ -> Colon(), D-i)
        sz = prob.size[i]
        (v_left, v_right) = a.bvs[i]
        (c_left, c_right) = prob.boundaries[i]
        a1_rev = view(a1, c1..., sz:-1:1, c2...)
        h = prob.step[i]
        g = prob.grid[i]
        v_left !== nothing && add_boundary_term_4th_order!(a1, v_left, c1, c2, h, c_left, g, false)
        v_right !== nothing && add_boundary_term_4th_order!(a1_rev, v_right, c1, c2, h, c_right, g, true)
    end
    solve(prob, a1)
end

function add_boundary_term_4th_order!(b, values, c1, c2, h, ::Dirichlet, ::Nothing, _is_right)
    @. b[c1..., 1:1, c2...] -= (5/6h^2) * values
    @. b[c1..., 2:2, c2...] += (1/12h^2) * values
end

function add_boundary_term_4th_order!(b, values, c1, c2, h, ::Dirichlet, ::Offset, _is_right)
    @. b[c1..., 1:1, c2...] -= (640/189h^2) * values
    @. b[c1..., 2:2, c2...] += (64/189h^2) * values
end

function add_boundary_term_4th_order!(b, values, c1, c2, h, ::Neumann, ::Nothing, is_right)
    r1 = ifelse(is_right, -1, 1) * (25/6h)
    r2 = ifelse(is_right, -1, 1) * (5/12h)
    @. b[c1..., 1:1, c2...] += r1 * values
    @. b[c1..., 2:2, c2...] -= r2 * values
end

function add_boundary_term_4th_order!(b, values, c1, c2, h, ::Neumann, ::Offset, is_right)
    r1 = ifelse(is_right, -1, 1) * (1600/1689h)
    r2 = ifelse(is_right, -1, 1) * (160/1689h)
    @. b[c1..., 1:1, c2...] += r1 * values
    @. b[c1..., 2:2, c2...] -= r2 * values
end

function mul_boundary!(b, a, c1, c2, h, ::Dirichlet, ::Nothing, _is_right)
    @. b[c1..., 1:1, c2...] -= ((7//6) * a[c1..., 1:1, c2...] +
        (-5//3) * a[c1..., 2:2, c2...] +
        (5//4) * a[c1..., 3:3, c2...] +
        (-1//2) * a[c1..., 4:4, c2...] +
        (1//12) * a[c1..., 5:5, c2...]) / h^2
end

function mul_boundary!(b, a, c1, c2, h, ::Dirichlet, ::Offset, _is_right)
    @. b[c1..., 1:1, c2...] -= ((-19//12) * a[c1..., 1:1, c2...] +
        (37//36) * a[c1..., 2:2, c2...] +
        (-5//12) * a[c1..., 3:3, c2...] +
        (2//21) * a[c1..., 4:4, c2...] +
        (-1/108) * a[c1..., 5:5, c2...]) / h^2

    @. b[c1..., 2:2, c2...] -= ((1//3) * a[c1..., 1:1, c2...] +
        (-5//18) * a[c1..., 2:2, c2...] +
        (1//6) * a[c1..., 3:3, c2...] +
        (-5//84) * a[c1..., 4:4, c2...] +
        (1//108) * a[c1..., 5:5, c2...]) / h^2
end

function mul_boundary!(b, a, c1, c2, h, ::Neumann, ::Nothing, is_right)
    @. b[c1..., 1:1, c2...] -= ((-235//72) * a[c1..., 1:1, c2...] +
        (16//3) * a[c1..., 2:2, c2...] +
        (-17//6) * a[c1..., 3:3, c2...] +
        (8//9) * a[c1..., 4:4, c2...] +
        (-1//8) * a[c1..., 5:5, c2...]) / h^2

    @. b[c1..., 2:2, c2...] -= ((65/144) * a[c1..., 1:1, c2...] +
        (-3//4) * a[c1..., 2:2, c2...] +
        (5//12) * a[c1..., 3:3, c2...] +
        (-5//36) * a[c1..., 4:4, c2...] +
        (1//48) * a[c1..., 5:5, c2...]) / h^2
end

function mul_boundary!(b, a, c1, c2, h, ::Neumann, ::Offset, is_right)
    @. b[c1..., 1:1, c2...] -= ((929//2252) * a[c1..., 1:1, c2...] +
        (-17791//20268) * a[c1..., 2:2, c2...] +
        (4745//6756) * a[c1..., 3:3, c2...] +
        (-482//1689) * a[c1..., 4:4, c2...] +
        (979//20268) * a[c1..., 5:5, c2...]) / h^2

    @. b[c1..., 2:2, c2...] -= ((19//563) * a[c1..., 1:1, c2...] +
        (-715//10134) * a[c1..., 2:2, c2...] +
        (185//3378) * a[c1..., 3:3, c2...] +
        (-145//6756) * a[c1..., 4:4, c2...] +
        (71//20268) * a[c1..., 5:5, c2...]) / h^2
end

function is_singular(prob::FourthOrder)
    for bc in prob.boundaries
        if bc[1] isa Dirichlet || bc[2] isa Dirichlet
            return false
        end
    end
    true
end
