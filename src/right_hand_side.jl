#=
The right-hand side of a Poisson equation consists of the values for Δu, as well as the values for u or du/dn on the
boundaries, depending on the different boundary conditions.
=#

struct WithBoundaries{T,N,BVS}
    rhs::Array{T,N}
    bvs::BVS
end

#=
Promotion of boundary values along a single axis:
- A single value or array is treated as being the values for both sides
- A tuple is treated as-is, i.e. (left values, right values)
=#
promote_boundary_value(T1, ::Nothing) = (nothing, nothing)
promote_boundary_value(T1, t::T) where {T<:Real} = (T1(t), T1(t))
promote_boundary_value(T1, t::Array{T,N}) where {T<:Real,N} = (T1.(t), T1.(t))
promote_boundary_value(T1, tup::NTuple{2,Union{Real,AbstractArray{<:Real}}}) = (T1.(tup[1]), T1.(tup[2]))

#=
Promotion of the tuple of boundary values
- The tuple can consist of one entry for each axis, in which case we promote them all individually
- It can also be a single entry, in which case it's repeated
=#
promote_boundary_values(T, ::Val{N}, bcs::NTuple{N,Any}) where {N} = promote_boundary_value.(T, bcs)
promote_boundary_values(T1, ndims::Val{N}, v::Tuple{TUP}) where {T,N,M,TUP<:Union{Nothing,T,Array{T,M}}} = begin
    _v = promote_boundary_value(T1, v[1])
    ntuple(_ -> _v, ndims)
end

function with_boundaries(rhs::Array{T,N}, boundaries...) where {T,N}
    ndims = Val(N)
    _boundaries = promote_boundary_values(T, ndims, boundaries)
    WithBoundaries(rhs, _boundaries)
end
