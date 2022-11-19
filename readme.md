# Fourier Poisson solvers

Methods to solve the Poisson equation up to second-order accuracy. These work using the Fourier-like transforms from
FFTW.jl, meaning they run in `N log N` time for a problem consisting of `N` data.

Second-order accuracy is possible since the matrix that one obtains when discretising a Poisson equation with a second-
order finite difference scheme is diagonalised by a Discrete Sine/Cosine/Hartley transform, meaning that we can compute
`A \ b` for any `b` in `N log N` time without using a linear solver: we simply use a transform to write `b` in terms
of the eigenvectors of `A`, divide those coefficients by the eigenvalues of `A`, and then use a (possibly different)
transform to convert back from eigenvector coefficients to the result.

This can be done for any combination of Dirichlet and Neumann boundary conditions, and can be made to work for normal
grids (where the boundaries are on a grid point) and offset grids (where the boundaries are halfway in between two grid
points).

One downside of this method is that while it generalises to higher dimensions, it does not generalise to higher orders
of accuracy: higher-order accurate discretisation of the Poisson equation result in matrices that are *not* diagonalised
by any Fourier transform, meaning we can't solve those linear systems of equations with such a transform. Fourth-order
compact schemes, in which the discretisation is kept the same but a correction term is added to the right-hand side,
*can* work, but only on uniform grids (i.e. when Δx=Δy=...), and their implementation becomes increasingly complex in
higher dimensions. As such, these fourth-order methods are not implemented here.

Right now, the work that has gone into this code is mostly in figuring out which transforms to use in which cases and
what the corresponding eigenvalues are that I should divide by. All of these things depend on the boundary conditions
and grid type, of which there are 9 combinations:

- Periodic boundary conditions (grid type is not relevant as a result)
- Dirichlet or Neumann boundary conditions on left boundary (x2), as well as on the right boundary (x2), for normal or
  offset grids (x2), so 8 combinations with non-periodic boundary conditions

## TODO

- Test
  - Test order of accuracy on 1D, 2D(, 3D) problems
  - Can make more tests based on solutions, then computing their Laplacian with ForwardDiff
  - Test API
- Extend
  - Can I use multiple FFT backends? Right now only `FFTW.jl` is supported and the `AbstractFFTs.jl` interface does not
    include support for DSTs/DCTs, but I could in theory write them myself in a way that just falls back to an FFT on a
    padded array. That would be inefficient though, but might be useful depending on how ubiquitous FFTW is.
  - Is it possible to extend to Robin boundary conditions? It doesn't look like it, but if it is then that would be very
    useful for i.e. advection-diffusion equations after a change of variables.
  - It is definitely possible to extend to the Helmholtz equation, although I'm not sure how useful that is
  - It should be possible to use the 2nd order accurate methods to create 4th-order iterative methods by matrix splitting
- Compare performance to multi-grid methods
- Organise
- Polish
  - Implement `display` and `show` for `PoissonProblem` to just give an overview instead of printing everything including coefficients and FFTW plans
  - Add rigorous checks in constructors so that as many errors as possible (regarding array dimensions etc.) are caught immediately
  - Make sure I only define public functions on my own types, i.e. no piracy
  - Maybe don't artificially limit the periodic boundary condition case to 2nd order accuracy
    - Doing this would require more specific under-relaxation parameter choices in the fourth-order algorithms, though
- Make public, perhaps publish on JuliaHub?
  - Would definitely require adding proper documentation
