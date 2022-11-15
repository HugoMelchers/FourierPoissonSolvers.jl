#=
For an axis with Dirichlet boundary conditions on both sides, we use a Discrete Sine Transform (DST).
=#

# Dirichlet boundary conditions on a normal grid

fwd_transform(::Dirichlet, ::Dirichlet, ::Nothing) = RODFT00
bwd_transform(::Dirichlet, ::Dirichlet, ::Nothing) = RODFT00
scalingfactor(::Dirichlet, ::Dirichlet, ::Nothing, n) = 2n + 2
frequencies(::Dirichlet, ::Dirichlet, ::Nothing, n) = LinRange(0, π, n+2)[2:end-1]

function update_left_boundary!(correction, pitch, bc::Dirichlet, ::Nothing)
    correction .-= bc.values .* (1 / pitch^2)
end

function update_right_boundary!(correction, pitch, bc::Dirichlet, ::Nothing)
    correction .-= bc.values .* (1 / pitch^2)
end

# Dirichlet boundary conditions on an offset grid

fwd_transform(::Dirichlet, ::Dirichlet, ::Offset) = RODFT10
bwd_transform(::Dirichlet, ::Dirichlet, ::Offset) = RODFT01
scalingfactor(::Dirichlet, ::Dirichlet, ::Offset, n) = 2n
frequencies(::Dirichlet, ::Dirichlet, ::Offset, n) = LinRange(0, π, n+1)[2:end]

#=
Note that the `update_*_boundary!` methods for Dirichlet boundary conditions with offset grids are the only such methods
that also add a correction term based on the right hand side of the equation, rather than only the boundary values. The
reason for this is that in that case, the correct approximation for u''(h/2) given u(0), u(h/2), and u(3h/2), is

    [ (8/3)u(0) - 4u(h/2) + (4/3)u(3h/2) ] / h² = u"(h/2) + 𝒪(h),

meaning we obtain the equation

    [ -4u(h/2) + (4/3)u(3h/2) ] / h² = u"(h/2) + 8u(0)/3h² + 𝒪(h),

but the DST corresponds to a matrix whose top row is [-3, 1, (zeros)], which is the above equation multiplied by 3/4:

    [ -3u(h/2) + u(3h/2) ] / h² = (3/4)u"(h/2) + 2u(0)/h² + 𝒪(h).

So, for better accuracy we should subtract ¼u"(h/2) from the right-hand side. Note that the solution is actually second-
order accurate regardless of whether this correction is added. However, without the correction the solution will not be
exact for problems where the true solution is a polynomial of degree 2, and will generally have an error that is greater
by some constant factor that depends on the problem.

Unfortunately, this reasoning only works for 1D problems. In two or more dimensions, the right hand side is not just the
second derivative of `u` in the desired axis, but the sum of second derivatives over all axes. That meanst that we can't
simply add such a correction term to the right-hand side as it requires knowing the individual second derivatives of
`u`, instead of their sum.
=#

function update_left_boundary!(correction, pitch, bc::Dirichlet, ::Offset)
    correction .-= bc.values .* (2 / pitch^2)
end

function update_right_boundary!(correction, pitch, bc::Dirichlet, ::Offset)
    correction .-= bc.values .* (2 / pitch^2)
end
