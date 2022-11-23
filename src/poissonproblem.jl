node_range(_, _, ::Offset, n) = (2n + 1, 2:2:(2n))
node_range(::Periodic, ::Periodic, ::Nothing, n) = (2n + 1, 2:2:(2n))
node_range(left::Union{Dirichlet,Neumann}, right::Union{Dirichlet,Neumann}, ::Nothing, n) = begin
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
struct PoissonProblem{T,N,BCS,G,Plan1,Plan2}
    size::NTuple{N,Int}
    step::NTuple{N,T}
    boundaries::BCS
    grid::G
    lims::NTuple{N,NTuple{2,T}}
    coefficients::Array{T,N}
    fwd_plan::r2rFFTWPlan{T,Plan1,true,N,UnitRange{Int}}
    bwd_plan::r2rFFTWPlan{T,Plan2,true,N,UnitRange{Int}}
end

promote_boundary_condition(bc::Union{Dirichlet,Neumann,Periodic}) = (bc, bc)
promote_boundary_condition(bc::NTuple{2,Union{Dirichlet,Neumann,Periodic}}) = bc
promote_boundary_conditions(::Val{N}, bcs::NTuple{N,Union{Dirichlet,Neumann,Periodic,NTuple{2,Union{Dirichlet,Neumann,Periodic}}}}) where {N} = promote_boundary_condition.(bcs)
promote_boundary_conditions(ndims::Val{N}, bcs::Union{Dirichlet,Neumann,Periodic}) where {N} = ntuple(_ -> (bcs, bcs), ndims)

#=
TODO: think about whether to keep this method. Creating a PoissonProblem with 

    boundaries=(Dirichlet(), Neumann())

to have mixed boundary conditions along each axis seems like a fairly uncommon use case, and makes the 2-dimensional
case, which is pretty common, an exception to an otherwise not super useful rule.
=#
promote_boundary_conditions(ndims::Val{N}, bc::NTuple{2,Union{Dirichlet,Neumann,Periodic}}) where {N} = begin
    promoted = promote_boundary_condition(bc)
    ntuple(_ -> promoted, ndims)
end

promote_boundary_conditions(::Val{2}, bcs::NTuple{2,Union{Dirichlet,Neumann,Periodic}}) = ((bcs[1], bcs[1]), (bcs[2], bcs[2]))
promote_lim(T, lims::NTuple{2,Real}) = T.(lims)
promote_lims(T, ::Val{N}, lims::NTuple{N,NTuple{2,Real}}) where {N} = promote_lim.(T, lims)
promote_lims(T, ndims::Val{N}, lims::NTuple{2,Real}) where {N} = begin
    promoted = promote_lim(T, lims)
    ntuple(_ -> promoted, ndims)
end
promote_grid(ndims::Val{N}, grid::Union{Nothing,Offset}) where {N} = ntuple(_ -> grid, ndims)
promote_grid(::Val{N}, grid::NTuple{N,Union{Nothing,Offset}}) where {N} = grid

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


gettype(lims::Tuple{<:Real, <:Real}) = widestfloat(floattype(typeof(lims[1])), floattype(typeof(lims[2])))
gettype(lims) = reduce(widestfloat, gettype.(lims))

"""
    PoissonProblem(size::NTuple{N,Int}; boundaries, lims, grid=nothing) where {N}

Creates a `PoissonProblem` over `N` dimensions, where the number of grid points along the `k`-th axis is `size[k]`.
Keyword arguments:

- `boundaries`: the types of boundary conditions of the problem. Can be specified in the following ways:
  - `boundaries=bc`, with `bc` one of `Periodic()`, `Dirichlet()`, or `Neumann()` sets the boundary condition to that
    type for all `N` dimensions, and for both sides in that dimension
  - `boundaries=(bc1, bc2, ..., bcN)` sets the boundary condition to `bc1` on both sides for the first dimension, `bc2`
    on both sides for the second dimension, and so on
  - `boundaries=((bc1left, bc1right), (bc2left, bc2right), ...)` allow setting boundary conditions individually for the
    left and right boundaries for each dimension
  The last two of these ways can be mixed and matched. For example, `boundaries=(Periodic(), (Dirichlet(),
  Neumann()))` for a 2-dimensional problem where the first dimension has periodic boundary conditions and the second
  dimension has Dirichlet boundary conditions on one side and Neumann on the other. Note that `Periodic` boundary
  conditions always apply to both sides, i.e. setting the left boundary to periodic but the right boundary to
  Dirichlet or Neumann does not work.

- `lims`: the lower and upper limits of the interval(s), at which the boundary conditions apply:
  - `lims=(x1, x2)` sets the interval to `[x1, x2]` along each dimension
  - `lims=((x1, x2), (y1, y2), ...)` sets the interval along each dimension to the corresponding pair

`grid`: determines whether the grid points are 'normal' or 'offset' (staggered). For a grid with step `h` between
  the grid points, a normal grid means that the end point of the interval lies on a grid point (for Dirichlet boundary
  conditions), or a distance `h` after the first/last grid point (for Neumann boundary conditions). An offset grid
  means that the boundaries lie `h/2` past the first/last point, independent of the boundary conditions.
  - `grid=nothing` (default) sets the grid to normal along each dimension
  - `grid=Offset()` sets the grid to offset along each dimension
  - `grid=(g1, g2, ..., gN)` sets the grid to normal or offset along each dimension separately
"""
function PoissonProblem(size::NTuple{N,Int}, T=nothing; boundaries, lims, grid=nothing) where {N}
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
    coefficients = eigenvalues.(left_boundaries, right_boundaries, _grid, size) ./ _step .^ 2
    _coefficients = zeros(_T, size...)
    transforms_fwd = fwd_transform.(left_boundaries, right_boundaries, _grid)
    transforms_bwd = bwd_transform.(left_boundaries, right_boundaries, _grid)
    fwd_plan = plan_r2r!(_coefficients, transforms_fwd)
    bwd_plan = plan_r2r!(_coefficients, transforms_bwd)
    for i in 1:length(size)
        _coefficients .+= reshape(coefficients[i], ntuple(_ -> 1, i - 1)..., :)
    end
    _coefficients .*= prod(scalingfactor.(left_boundaries, right_boundaries, _grid, size))
    _coefficients .= 1 ./ _coefficients
    prob = PoissonProblem(size, _step, _boundaries, _grid, _lims, _coefficients, fwd_plan, bwd_plan)
    is_singular(prob) && (prob.coefficients[1] = 0.0)
    prob
end

function add_boundary_terms!(f, i, step, bc, values, grid)
    #=
    The Poisson equation `Δu=f` with boundary conditions is discretised into a linear system `Au=f+Δf` where `Δf` is an
    additional term that corrects for the boundary conditions. This function adds that correction to `f`.
    =#
    bc[1] isa Periodic && return
    D = ndims(f)
    c1 = ntuple(_ -> Colon(), i-1)
    c2 = ntuple(_ -> Colon(), D-i)
    sz = size(f, i)
    view_left = view(f, c1..., 1:1, c2...)
    view_right = view(f, c1..., sz:sz, c2...)
    values[1] !== nothing && add_boundary_term!(view_left, step, bc[1], values[1], grid, false)
    values[2] !== nothing && add_boundary_term!(view_right, step, bc[2], values[2], grid, true)
end
