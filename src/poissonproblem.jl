# A struct so that creating an axis that is offset can be done by adding an `Offset()` argument, instead of a less clear `true`
struct Offset end

# Periodic boundary conditions, which require no additional data
struct Periodic end

# Dirichlet boundary conditions with the given values. Can be either nothing, a constant, or an array
struct Dirichlet end

# Neumann boundary conditions with the given values. Can be either nothing, a constant, or an array.
# Note that right now, the values are interpreted as partial derivatives, rather than normal derivatives.
# This means that compared to the normal derivative formulation, the signs are swapped one of the two boundaries.
struct Neumann end

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

struct PoissonProblem{T,N,BCS,G,Plan1,Plan2}
    size::NTuple{N,Int}
    step::NTuple{N,T}
    boundaries::BCS
    grid::G
    lims::NTuple{N,NTuple{2,T}}
    nodes::NTuple{N,StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T},Int}}
    coefficients::Array{T,N}
    fwd_plan::r2rFFTWPlan{T,Plan1,true,N,UnitRange{Int}}
    bwd_plan::r2rFFTWPlan{T,Plan2,true,N,UnitRange{Int}}
end

promote_boundary_condition(bc::Union{Dirichlet,Neumann,Periodic}) = (bc, bc)
promote_boundary_condition(bc::NTuple{2,Union{Dirichlet,Neumann,Periodic}}) = bc
promote_boundary_conditions(::Val{N}, bcs::NTuple{N,Union{Dirichlet,Neumann,Periodic,NTuple{2,Union{Dirichlet,Neumann,Periodic}}}}) where {N} = promote_boundary_condition.(bcs)
promote_boundary_conditions(ndims::Val{N}, bcs::Union{Dirichlet,Neumann,Periodic}) where {N} = ntuple(_ -> (bcs, bcs), ndims)
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

"""
    PoissonProblem(size; boundaries, lims, grid)
"""
function PoissonProblem(size::NTuple{N,Int}; boundaries, lims, grid) where {N}
    # 'promote' grid kind and boundary condition types so that these arguments don't have to be repeated if they are
    # identical for all axes
    T = Float64 # TODO determine from input
    _grid = promote_grid(Val(N), grid)
    _boundaries = promote_boundary_conditions(Val(N), boundaries)
    _lims = promote_lims(T, Val(N), lims)
    left_boundaries = getindex.(_boundaries, 1)
    right_boundaries = getindex.(_boundaries, 2)
    left_lims = getindex.(_lims, 1)
    right_lims = getindex.(_lims, 2)
    _nodes = nodes.(left_boundaries, right_boundaries, _grid, size, left_lims, right_lims)
    _step = step.(left_boundaries, right_boundaries, _grid, size, left_lims, right_lims)
    coefficients = frequencies.(left_boundaries, right_boundaries, _grid, size) ./ _step .^ 2
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
    prob = PoissonProblem(size, _step, _boundaries, _grid, _lims, _nodes, _coefficients, fwd_plan, bwd_plan)
    is_singular(prob) && (prob.coefficients[1] = 0.0)
    prob
end
