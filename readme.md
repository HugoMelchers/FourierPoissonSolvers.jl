# Fourier Poisson solvers

Methods to solve the Poisson equation up to second-order accuracy. These work using the Fourier-like transforms from
FFTW.jl, meaning they run in `N log N` time for a problem consisting of `N` data.

## TODO

- Finish handling of periodic boundary conditions
  - Use more accurate eigenvalues in this case
  - Make sure the fourth-order methods don't iterate if all boundary conditions are periodic
- Finish fourth-order methods
  - Test
  - Check that everything iterates until convergence
  - See if I can make it iterate in the spectral domain, so that the only DFTs needed are at the start/end
- Add `Float32` support
  - Mostly involves finding the widest floating-point type in the constructor arguments, and defaulting to `Float64` if
    there are none
  - Also make sure I do fewer iterations in the fourth-order methods
- Test
  - Test order of accuracy on 1D, 2D(, 3D) problems
  - Can make more tests based on solutions, then computing their Laplacian with ForwardDiff
  - Test API
    - Allow specifying mixed boundary conditions for all axes as `boundaries=((Dirichlet(), Neumann()),)`
- Extend
  - Can I use multiple FFT backends? Right now only `FFTW.jl` is supported and the `AbstractFFTs.jl` interface does not
    include support for DSTs/DCTs, but I could in theory write them myself in a way that just falls back to an FFT on a
    padded array. That would be inefficient though, but might be useful depending on how ubiquitous FFTW is.
  - Is it possible to extend to Robin boundary conditions? It doesn't look like it, but if it is then that would be very
    useful for i.e. advection-diffusion equations after a change of variables.
  - It is definitely possible to extend to the Helmholtz equation, although I'm not sure how useful that is
- Compare performance to multi-grid methods
- Organise
- Polish
  - Implement `display` and `show` for `PoissonProblem` to just give an overview instead of printing everything including coefficients and FFTW plans
  - Add rigorous checks in constructors so that as many errors as possible (regarding array dimensions etc.) are caught immediately
  - Make sure I only define public functions on my own types, i.e. no piracy
- Make public, perhaps publish on JuliaHub?
  - Would definitely require adding proper documentation
