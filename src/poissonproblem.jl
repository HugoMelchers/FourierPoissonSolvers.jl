node_range(_, _, ::Offset, n) = (2n + 1, 2:2:(2n))

node_range(::Periodic, ::Periodic, ::Nothing, n) = (2n + 1, 2:2:(2n))

function node_range(left::AperiodicBC, right::AperiodicBC, ::Nothing, n)
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
    range = (1+skip_left):(logical_size-skip_right)
    (logical_size, range)
end

function _nodes(left, right, grid, n, x1, x2)
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

promote_boundary_condition(bc::BoundaryCondition) = (bc, bc)
promote_boundary_condition(bc::BCPair) = bc
promote_boundary_conditions(::Val{N}, bcs::NTuple{N,Union{BoundaryCondition,BCPair}}) where {N} = promote_boundary_condition.(bcs)
promote_boundary_conditions(ndims::Val{N}, bcs::BoundaryCondition) where {N} = ntuple(_ -> (bcs, bcs), ndims)
promote_boundary_conditions(ndims::Val{N}, bcs::Tuple{BCPair}) where {N} = ntuple(_ -> bcs[1], ndims)
promote_boundary_conditions(::Val{1}, bcs::BCPair) = (bcs,)

promote_lim(T, lims::NTuple{2,Real}) = T.(lims)
promote_lims(T, ::Val{N}, lims::NTuple{N,NTuple{2,Real}}) where {N} = promote_lim.(T, lims)
promote_lims(T, ndims::Val{N}, lims::NTuple{2,Real}) where {N} = begin
    promoted = promote_lim(T, lims)
    ntuple(_ -> promoted, ndims)
end
promote_grid(ndims::Val{N}, grid::AnyGrid) where {N} = ntuple(_ -> grid, ndims)
promote_grid(::Val{N}, grid::NTuple{N,AnyGrid}) where {N} = grid

floattype(::Type{Float32}) = Float32
floattype(::Type{Float64}) = Float64
floattype(::Type{<:Real}) = Nothing

widestfloat(::Type{Float32}, ::Type{Float32}) = Float32
widestfloat(::Type{Float32}, ::Type{Float64}) = Float64
widestfloat(::Type{Float64}, ::Type{Float32}) = Float64
widestfloat(::Type{Float64}, ::Type{Float64}) = Float64
widestfloat(::Type{Nothing}, ::Type{Float32}) = Float32
widestfloat(::Type{Nothing}, ::Type{Float64}) = Float64
widestfloat(::Type{Float32}, ::Type{Nothing}) = Float32
widestfloat(::Type{Float64}, ::Type{Nothing}) = Float64
widestfloat(::Type{Nothing}, ::Type{Nothing}) = Nothing


gettype(lims::Tuple{<:Real,<:Real}) = widestfloat(floattype(typeof(lims[1])), floattype(typeof(lims[2])))
gettype(lims) = reduce(widestfloat, gettype.(lims))

function spectral_interval(::Type{T}, boundaries, grids) where {T}
    r1 = zero(T)
    r2 = zero(T)
    radius(::Dirichlet, ::Nothing) = 0.2577122354265
    radius(::Dirichlet, ::Offset) = -0.38005393284828
    radius(::Neumann, ::Nothing) = -0.857024488446
    radius(::Neumann, ::Offset) = 0.09013067273
    radius(::Periodic, _) = 0.0
    for i in 1:length(grids)
        (left, right) = boundaries[i]
        grid = grids[i]
        r_left = T(radius(left, grid))
        r_right = T(radius(right, grid))
        r1 = min(r1, r_left, r_right)
        r2 = max(r2, r_left, r_right)
    end
    r1, r2
end

function optimal_relaxation_parameters(::Type{T}, a, b, k) where {T}
    #=
    Computes the optimal under-relaxation parameters when doing fixed-point iteration `x <- x - A*x`, given that all
    eigenvalues of `A` lie in the interval `[a, b]`. The optimal under-relaxation factors were derived by T. Wadayama
    and S. Takabe in https://arxiv.org/abs/2001.03280.
    =#
    [inv((a + b) / 2 + (b - a) / 2 * cos(x)) for x in LinRange(0, T(pi), 2k + 1)[2:2:end]]
end

function steps_to_error(::Type{T}, a, b, r) where {T}
    #=
    Computes the number of iterations required to reach a desired error `r`. The error factor after `k` iterations of
    under- relaxation with optimally chosen under-relaxation parameters can be expressed as
    `sech(k acosh((b+a) / (b-a)))`, which is derived by Takabe and Wadayama in https://arxiv.org/pdf/2010.13335,
    appendix C. By requiring this expression to be at most `r`, we can derive an expression for `k` resulting in the
    computation below.
    =#
    p = (b + a) / (b - a)
    ceil(Int, asech(r) / acosh(p))
end

function relaxation_parameters(::Type{T}, boundaries, grids) where {T}
    r1, r2 = spectral_interval(T, boundaries, grids)
    a, b = 1 - T(r2), 1 - T(r1)
    k = steps_to_error(T, a, b, eps(T)) + 3
    optimal_relaxation_parameters(T, a, b, k)
end

struct PoissonProblem{T,N,Order,BCS,G,Plan1,Plan2,R}
    size::NTuple{N,Int}
    step::NTuple{N,T}
    boundaries::BCS
    grid::G
    lims::NTuple{N,NTuple{2,T}}
    coefficients::Array{T,N}
    fwd_plan::r2rFFTWPlan{T,Plan1,true,N,UnitRange{Int}}
    bwd_plan::r2rFFTWPlan{T,Plan2,true,N,UnitRange{Int}}
    order::Order
    relaxationparameters::R
end

"""
    PoissonProblem(size::NTuple{N,Int}, T=nothing; boundaries, lims, grid=nothing, order=Val(2), fftw_flags=FFTW_ESTIMATE, fftw_timelimit=Inf) where {N}

Creates a `PoissonProblem` over `N` dimensions with element type `T` (either `Float32` or `Float64`), where the number
of grid points along the `k`-th axis is `size[k]`. Keyword arguments:

- `boundaries`: the types of boundary conditions of the problem. Can be specified in the following ways:
  - `boundaries=bc`, with `bc` one of `Periodic()`, `Dirichlet()`, or `Neumann()` sets the boundary condition to that
    type for all `N` dimensions, and for both sides in each dimension
  - `boundaries=((bc_left, bc_right),)` sets all left boundary conditions to `bc_left` and all right boundary conditions
    to `bc_right`
  - `boundaries=(bc1, bc2, ..., bcN)` sets the boundary condition to `bc1` on both sides for the first dimension, `bc2`
    on both sides for the second dimension, and so on
  - `boundaries=((bc1left, bc1right), (bc2left, bc2right), ...)` sets the boundary conditions individually for the left
    and right boundaries for each dimension

  The last two of these ways can be mixed and matched. For example, `boundaries=(Periodic(), (Dirichlet(),
  Neumann()))` for a 2-dimensional problem where the first dimension has periodic boundary conditions and the second
  dimension has Dirichlet boundary conditions on one side and Neumann on the other. Note that `Periodic` boundary
  conditions always apply to both sides, i.e. setting the left boundary to periodic but the right boundary to
  Dirichlet or Neumann does not work.

- `lims`: the lower and upper limits of the interval(s), at which the boundary conditions apply:
  - `lims=(x1, x2)` sets the interval to `[x1, x2]` along each dimension
  - `lims=((x1, x2), (y1, y2), ...)` sets the interval along each dimension to the corresponding pair

- `grid`: determines whether the grid points are 'normal' or 'offset' (staggered). For a grid with step `h` between
  the grid points, a normal grid means that the end point of the interval lies on a grid point (for Neumann boundary
  conditions), or a distance `h` after the first/last grid point (for Dirichlet boundary conditions). An offset grid
  means that the boundaries lie `h/2` past the first/last point, independent of the boundary conditions.
  - `grid=nothing` (default) sets the grid to normal along each dimension
  - `grid=Offset()` sets the grid to offset along each dimension
  - `grid=(g1, g2, ..., gN)` sets the grid to normal or offset along each dimension separately

- `order`: can be set to `Val(2)` (default) or `Val(4)`, to choose whether to solve the equation to second-order or
  fourth-order accuracy. Note that while the second-order accurate methods are direct, the fourth-order methods are
  iterative and may require several dozen iterations, in which case they are also several dozen times slower than
  a second-order problem of the same size. As such, using second-order methods is recommended.

- `fftw_flags` and `fftw_timelimit` allow specifying keyword arguments to `FFTW.plan_r2r!`, which finds efficient
  plans for the transforms that are used. For possible values for these arguments, see the `AbstractFFTs` documentation
  <https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.plan_fft>. The `FFTW.{PATIENT, MEAUSRE, ...}`
  symbols are re-exported as `FFTW_PATIENT`, `FFTW_MEASURE`, and so on.
"""
function PoissonProblem(size::NTuple{N,Int}, T=nothing; boundaries, lims, grid=nothing, order=Val(2), fftw_flags=FFTW_ESTIMATE, fftw_timelimit=Inf) where {N}
    _T = if T !== nothing
        T
    else
        T_inferred = gettype(lims)
        if T_inferred !== Nothing
            T_inferred
        else
            Float64
        end
    end
    _grid = promote_grid(Val(N), grid)
    _boundaries = promote_boundary_conditions(Val(N), boundaries)
    _lims = promote_lims(_T, Val(N), lims)
    left_boundaries = getindex.(_boundaries, 1)
    right_boundaries = getindex.(_boundaries, 2)
    left_lims = getindex.(_lims, 1)
    right_lims = getindex.(_lims, 2)
    _step = step.(left_boundaries, right_boundaries, _grid, size, left_lims, right_lims)
    coefficients = if order == Val(2)
        eigenvalues2.(left_boundaries, right_boundaries, _grid, size) ./ _step .^ 2
    else
        eigenvalues4.(left_boundaries, right_boundaries, _grid, size) ./ _step .^ 2
    end
    _coefficients = zeros(_T, size...)
    transforms_fwd = fwd_transform.(left_boundaries, right_boundaries, _grid)
    transforms_bwd = bwd_transform.(left_boundaries, right_boundaries, _grid)
    fwd_plan = plan_r2r!(_coefficients, transforms_fwd; flags=fftw_flags, timelimit=fftw_timelimit)
    bwd_plan = plan_r2r!(_coefficients, transforms_bwd; flags=fftw_flags, timelimit=fftw_timelimit)
    for i in 1:length(size)
        _coefficients .+= reshape(coefficients[i], ntuple(_ -> 1, i - 1)..., :)
    end
    _coefficients .*= prod(scalingfactor.(left_boundaries, right_boundaries, _grid, size))
    _coefficients .= 1 ./ _coefficients
    rs = if order == Val(4)
        relaxation_parameters(_T, _boundaries, _grid)
    else
        nothing
    end
    prob = PoissonProblem(size, _step, _boundaries, _grid, _lims, _coefficients, fwd_plan, bwd_plan, order, rs)
    is_singular(prob) && (prob.coefficients[1] = 0.0)
    prob
end

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
        if bc[1] isa Dirichlet || bc[2] isa Dirichlet
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
        elseif bc[1] isa Dirichlet || bc[2] isa Dirichlet
            return false
        end
    end
    true
end

"""
    fully_periodic(prob::PoissonProblem)

Returns true if this problem has periodic boundary conditions for all axes. If this is the case, the solution obtained
by `prob \\ rhs` is not second- or fourth-order accurate, but spectrally accurate, generally meaning a much lower error.
It also means that the second- and fourth-order methods both produce the same solution, and that the fourth-order method
obtains this solution in one iteration, unlike problems with other boundary conditions.
"""
function fully_periodic(prob::PoissonProblem)
    for bc in prob.boundaries
        if !(bc[1] isa Periodic && bc[2] isa Periodic)
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

"""
    nodes(prob::PoissonProblem)

Create a tuple of `range`s, corresponding to the grid points at which the solution is obtained.
"""
function nodes(prob::PoissonProblem)
    left_boundaries = getindex.(prob.boundaries, 1)
    right_boundaries = getindex.(prob.boundaries, 2)
    left_lims = getindex.(prob.lims, 1)
    right_lims = getindex.(prob.lims, 2)
    _nodes.(left_boundaries, right_boundaries, prob.grid, prob.size, left_lims, right_lims)
end
