#=
For an axis with Neumann boundary conditions on both sides, we use a Discrete Cosine Transform (DCT).
=#

fwd_transform(::Tuple{Neumann, Neumann}, ::Nothing) = REDFT00
bwd_transform(::Tuple{Neumann, Neumann}, ::Nothing) = REDFT00
scalingfactor(::Tuple{Neumann, Neumann}, ::Nothing, n) = 2n - 2
frequencies(::Tuple{Neumann, Neumann}, ::Nothing, n) = LinRange(0, π, n)

function update_left_boundary!(correction, pitch, bc::Neumann, ::Nothing)
    correction .+= bc.values .* (2 / pitch)
end

function update_right_boundary!(correction, pitch, bc::Neumann, ::Nothing)
    correction .-= bc.values .* (2 / pitch)
end

fwd_transform(::Tuple{Neumann, Neumann}, ::Offset) = REDFT10
bwd_transform(::Tuple{Neumann, Neumann}, ::Offset) = REDFT01
scalingfactor(::Tuple{Neumann, Neumann}, ::Offset, n) = 2n
frequencies(::Tuple{Neumann, Neumann}, ::Offset, n) = LinRange(0, π, n+1)[1:end-1]

function update_left_boundary!(correction, pitch, bc::Neumann, ::Offset)
    correction .+= bc.values .* (1 / pitch)
end

function update_right_boundary!(correction, pitch, bc::Neumann, ::Offset)
    correction .-= bc.values .* (1 / pitch)
end
