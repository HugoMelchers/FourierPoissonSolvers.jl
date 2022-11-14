# Spectral Poisson solvers

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
  - Test order of accuracy on 1D, 2D, 3D problems
  - Also test that error is close to machine precision when the right-hand side is a polynomial of low degree
  - Can make more tests based on solutions, then computing their Laplacian with ForwardDiff
- Extend
  - Can I use multiple FFT backends? Right now only `FFTW.jl` is supported and the `AbstractFFTs.jl` interface does not
    include support for DSTs/DCTs, but I could in theory write them myself in a way that just falls back to an FFT on a
    padded array. That would be inefficient though, but might be useful depending on how ubiquitous FFTW is.
  - Is it possible to extend to Robin boundary conditions? It doesn't look like it, but if it is then that would be very
    useful for i.e. advection-diffusion equations after a change of variables.
  - It is definitely possible to extend to the Helmholtz equation, although I'm not sure how useful that is
  - It should be possible to use the 2nd order accurate methods to create 4th-order iterative methods by matrix splitting
- Improve performance
  - In 1D, see if just using the Tri-Diagonal Matrix Algorithm works as well (in terms of both performance and accuracy/stability)
    - Actually, it's always possible to only transform over `D-1` dimensions, and use TDMA for the remaining one
  - Make a pre-planned version
    - From plan, the solve would be
      - Set boundary conditions
      - Set rhs
      - Solve in-place
  - Combine transforms over different axes into one
    - `FFTW.r2r!` allows setting a different transform type along each dimension, so I can collect the types into a
      tuple and then do a single transform. This will only help in 2 or more dimensions, though.
    - Compare performance of in-place vs out-of-place DCTs/DSTs
  - Make code strongly typed
    - This is a bit difficult to figure out, as it requires some arithmetic with type variables which isn't possible
    - An alternative is to require that the arrays are `N`-dimensional instead of `N-1`, with a singleton dimension at the end (or at the index of the axis)
      - I should be able to do this just in the constructors
    - Make specialised versions of the boundary condition structs if the boundary values are constant, or constantly zero
  - Compare performance to multi-grid methods
- Organise
  - Project isn't huge, but still good to split up into different files, with lots of documentation
  - Use multiple dispatch to handle the different boundary conditions and axis types more easily
- Polish
  - Figure out a clean API for defining axes, problems and so on
  - Find a neat way to be able to access/modify the right-hand side and boundary values after creating a problem
    - Maybe define a `PoissonOperator` (or `LaplaceOperator`?) that contains boundary conditions etc, and call with `ldiv(op, rhs)`
  - Right now, to create a problem I must create axes with boundary conditions, but to create the boundary conditions
    I must know where to sample my function which depends on the axes. This is especially annoying when testing the
    implementations with non-offset grids, since then the x values depend on the grid. So it would be more convenient
    to first create an object representing the range of values, and then a separate thing that includes the boundary
    conditions.
- Make public, perhaps publish on JuliaHub?
  - Requires a better name, since these methods do not have spectral accuracy
  - Would definitely require adding proper documentation

## How to handle boundary conditions

So for a single axis, we have either a single Periodic, or a pair of Dirichlet/Neumann, which can both have associated
data or not (in case of homogeneous boundary conditions I don't want the overhead of storing an array of zeros).

Alternatively: plan a solution and then call it with many right-hand sides?

- `prob = PoissonProblem(...)`
  - `prob` should now have modifiable arrays for the right-hand side and boundary condition values

Implementation: for now, focus on the planless implementation, then write separate planned ones.

Planless implementation:

- For now, I can just do compute everything on the fly, and compute individual transforms etc.

Planned implementation:

- A function that, given a boundary condition (pair) and a length, returns:
  - The forward and inverse transforms to use
  - The eigenvalues to scale by, including the scaling factor of the transform
- Then I can use the transform types to plan the forward and backward transforms and execute them with `mul!(rhs, plan, rhs)`
