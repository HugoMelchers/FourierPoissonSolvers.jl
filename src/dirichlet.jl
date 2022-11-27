# Dirichlet boundary conditions
struct Dirichlet end

#=
For an axis with Dirichlet boundary conditions on both sides, we use a Discrete Sine Transform (DST).
=#

fwd_transform(::Dirichlet, ::Dirichlet, ::Nothing) = RODFT00
bwd_transform(::Dirichlet, ::Dirichlet, ::Nothing) = RODFT00
scalingfactor(::Dirichlet, ::Dirichlet, ::Nothing, n) = 2n + 2
wavenumbers(::Dirichlet, ::Dirichlet, ::Nothing, n) = LinRange(0, π, n + 2)[2:end-1]
correction_coefficients(::Dirichlet, ::Nothing) = ((7 // 6, -5 // 3, 5 // 4, -1 // 2, 1 // 12), nothing)

function add_boundary_term!(correction, ::Dirichlet, ::Nothing, values, h)
    correction .-= values .* (1 / h^2)
end

function add_boundary_term_4th_order!(b1, b2, ::Dirichlet, ::Nothing, values, h)
    b1 .-= (5 / 6h^2) .* values
    b2 .+= (1 / 12h^2) .* values
end

# Dirichlet boundary conditions on an offset grid

fwd_transform(::Dirichlet, ::Dirichlet, ::Offset) = RODFT10
bwd_transform(::Dirichlet, ::Dirichlet, ::Offset) = RODFT01
scalingfactor(::Dirichlet, ::Dirichlet, ::Offset, n) = 2n
wavenumbers(::Dirichlet, ::Dirichlet, ::Offset, n) = LinRange(0, π, n + 1)[2:end]
correction_coefficients(::Dirichlet, ::Offset) = ((-19 // 12, 37 // 36, -5 // 12, 2 // 21, -1 / 108), (1 // 3, -5 // 18, 1 // 6, -5 // 84, 1 // 108))

#=
Note that a Poisson problem with Dirichlet boundary conditions on an offset grid is the only case in which the solution
is not correct up to numerical precision if the exact solution is given by a quadratic function (which is generally true
for second-order accurate methods). The reason for this is that in that case, the correct approximation for u''(h/2)
given u(0), u(h/2), and u(3h/2), is

    [ (8/3)u(0) - 4u(h/2) + (4/3)u(3h/2) ] / h² = u"(h/2) + 𝒪(h),

meaning we obtain the equation

    [ -4u(h/2) + (4/3)u(3h/2) ] / h² = u"(h/2) + 8u(0)/3h² + 𝒪(h),

but the closest DST to this is the RODFT10, which corresponds to a matrix whose top row is [-3, 1, (zeros)], which is
the above equation multiplied by 3/4:

    [ -3u(h/2) + u(3h/2) ] / h² = (3/4)u"(h/2) + 2u(0)/h² + 𝒪(h).

So, for better accuracy we should subtract ¼u"(h/2) from the right-hand side. Note that the solution is actually second-
order accurate regardless of whether this correction is added. However, without the correction the solution will not be
exact for problems where the true solution is a polynomial of degree 2, and will generally have an error that is greater
by some constant factor that depends on the problem.

Unfortunately, this reasoning only works for 1D problems. In two or more dimensions, the right hand side is not just the
second derivative of `u` in the desired axis, but the sum of second derivatives over all axes. That means that we can't
simply add such a correction term to the right-hand side as it requires knowing the individual second derivatives of
`u`, instead of their sum.
=#

function add_boundary_term!(correction, ::Dirichlet, ::Offset, values, h)
    correction .-= (2 / h^2) .* values
end

function add_boundary_term_4th_order!(b1, b2, ::Dirichlet, ::Offset, values, h)
    b1 .-= (640 / 189h^2) .* values
    b2 .+= (64 / 189h^2) .* values
end
