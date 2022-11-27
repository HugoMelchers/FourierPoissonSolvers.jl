# Fourier Poisson solvers

Methods to solve the Poisson equation up to second-order accuracy. These work using the Fourier-like transforms from
FFTW.jl, meaning they run in `N log N` time for a problem consisting of `N` data.

## Features

- Support for `Float32` and `Float64`
- Dirichlet, Neumann, and periodic boundary conditions
  - Including mixed boundary conditions, e.g. Dirichlet on one side and Neumann on the other
- Any number of dimensions: not limited to 1D, 2D, and 3D
- Second-order accurate direct solvers
- Fourth-order accurate iterative solvers
  - The fourth-order methods converge in at most 15 iterations for single-precision and in at most 28 iterations for
    double precision, meaning they take about 15 and 28 times as much time as solving the same problem to second-order
    accuracy.
- Spectral accuracy for problems with periodic boundary conditions

## Not supported

- Other floating point types such as `Float16` or `BigFloat`
- GPU arrays, distributed arrays, etc.
- Robin boundary conditions

Other floating point types and GPU arrays are not supported since support for them is not present in `FFTW`. Other
FFT implementations exist, but these generally do not support DCTs and DSTs, which are the transforms required to
efficiently solve Poisson equations with Dirichlet or Neumann boundary conditions.

## TODO

- Test
  - Test order of accuracy on 1D, 2D(, 3D) problems
  - Can make more tests based on solutions, then computing their Laplacian with ForwardDiff
  - Try to test fourth-order methods (error doesn't decrease as neatly as for second-order methods, so order of accuracy
    is harder to verify numerically)
  - Test API
- Extend
  - See if multi-threading can be implemented easily
  - Allow passing FFTW flags for more exhaustive plan searching
- Compare performance to multi-grid methods
- Organise
- Polish
  - Implement `display` and `show` for `PoissonProblem` to just give an overview instead of printing everything
    including coefficients and FFTW plans
  - Add rigorous checks in constructors so that as many errors as possible (regarding array dimensions etc.) are caught
    immediately and with a clear error message
  - Make sure I only define public functions on my own types, i.e. no piracy
- Make public, perhaps publish on JuliaHub?
  - Would definitely require adding proper documentation
