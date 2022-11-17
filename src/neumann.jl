#=
For an axis with Neumann boundary conditions on both sides, we use a Discrete Cosine Transform (DCT).
=#

fwd_transform(::Neumann, ::Neumann, ::Nothing) = REDFT00
bwd_transform(::Neumann, ::Neumann, ::Nothing) = REDFT00
scalingfactor(::Neumann, ::Neumann, ::Nothing, n) = 2n - 2
frequencies(::Neumann, ::Neumann, ::Nothing, n) = frequency_response.(LinRange(0, π, n))

function update_left_boundary!(correction, pitch, ::Neumann, values, ::Nothing)
    correction .+= values .* (2 / pitch)
end

function update_right_boundary!(correction, pitch, ::Neumann, values, ::Nothing)
    correction .-= values .* (2 / pitch)
end

fwd_transform(::Neumann, ::Neumann, ::Offset) = REDFT10
bwd_transform(::Neumann, ::Neumann, ::Offset) = REDFT01
scalingfactor(::Neumann, ::Neumann, ::Offset, n) = 2n
frequencies(::Neumann, ::Neumann, ::Offset, n) = frequency_response.(LinRange(0, π, n+1)[1:end-1])

function update_left_boundary!(correction, pitch, ::Neumann, values, ::Offset)
    correction .+= values .* (1 / pitch)
end

function update_right_boundary!(correction, pitch, ::Neumann, values, ::Offset)
    correction .-= values .* (1 / pitch)
end
