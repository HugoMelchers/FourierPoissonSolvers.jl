#=
For an axis with periodic boundary conditions, it is normal to use a Fast Fourier Transform (FFT). However, this
transform has the disadvantage that the resulting values are complex, even under the assumption that the input data
is real. This also means we can't perform the transform in place, since we must allocate a complex array. Instead, for
periodic boundary conditions I use a Discrete Hartley Transform (DHT), which also assumes periodic data, but writes
it as a sum of sine and cosine waves instead of complex exponentials like a Fourier transform. This has the result of
always producing real outputs for real inputs, while still being done in `N log N` time, and still being easy to invert
(in fact, the DHT is its own inverse, up to a scaling factor). Using DHTs also has the advantage of fitting into
FFTW.jl's API more easily, since we can perform any real-to-real (r2r) transform with the same function call, just
by varying one of the input parameters. Furthermore, FFTW lets us transform a multidimensional array with different
transforms over different axes, which is not possible if one of the transforms we want to do is an FFT.
=#

fwd_transform(::Periodic, ::Periodic, _) = DHT
bwd_transform(::Periodic, ::Periodic, _) = DHT

scalingfactor(::Periodic, ::Periodic, _, n) = n

# TODO: maybe just use the more accurate eigenvalues for periodic boundary conditions
frequencies(::Periodic, ::Periodic, _, n) = 2pi .* fftfreq(n)
