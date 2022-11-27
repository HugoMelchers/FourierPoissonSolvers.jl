struct Neumann end

#=
For an axis with Neumann boundary conditions on both sides, we use a Discrete Cosine Transform (DCT).
=#

fwd_transform(::Neumann, ::Neumann, ::Nothing) = REDFT00
bwd_transform(::Neumann, ::Neumann, ::Nothing) = REDFT00
scalingfactor(::Neumann, ::Neumann, ::Nothing, n) = 2n - 2
wavenumbers(::Neumann, ::Neumann, ::Nothing, n) = LinRange(0, π, n)
correction_coefficients(::Neumann, ::Nothing) = ((-235 // 72, 16 // 3, -17 // 6, 8 // 9, -1 // 8), (65 // 144, -3 // 4, 5 // 12, -5 // 36, 1 // 48))

function add_boundary_term!(correction, ::Neumann, ::Nothing, values, h)
    scale = 2 / h
    correction .+= values .* scale
end

function add_boundary_term_4th_order!(b1, b2, ::Neumann, ::Nothing, values, h)
    r1 = 25 / 6h
    r2 = 5 / 12h
    b1 .+= r1 .* values
    b2 .-= r2 .* values
end

fwd_transform(::Neumann, ::Neumann, ::Offset) = REDFT10
bwd_transform(::Neumann, ::Neumann, ::Offset) = REDFT01
scalingfactor(::Neumann, ::Neumann, ::Offset, n) = 2n
wavenumbers(::Neumann, ::Neumann, ::Offset, n) = LinRange(0, π, n + 1)[1:end-1]
correction_coefficients(::Neumann, ::Offset) = ((929 // 2252, -17791 // 20268, 4745 // 6756, -482 // 1689, 979 // 20268), (19 // 563, -715 // 10134, 185 // 3378, -145 // 6756, 71 // 20268))

function add_boundary_term!(correction, ::Neumann, ::Offset, values, h)
    scale = 1 / h
    correction .+= values .* scale
end

function add_boundary_term_4th_order!(b1, b2, ::Neumann, ::Offset, values, h)
    r1 = 1600 / 1689h
    r2 = 160 / 1689h
    b1 .+= r1 .* values
    b2 .-= r2 .* values
end
