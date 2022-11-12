# Spectral Poisson solvers

## TODO

- Test
	- Test order of accuracy on 1D, 2D, 3D problems
	- Can just make tons of tests based on solutions, then computing their Laplacian with ForwardDiff
- Make a pre-planned version
	- From plan, the solve would be 
		- Set boundary conditions
		- Set rhs
		- Solve in-place
- Improve performance
	- Combine transforms over different axes into one
		- `FFTW.r2r!` allows setting a different transform type along each dimension, so I can collect the types into a
			tuple and then do a single transform. This will only help in 2 or more dimensions, though.
		- Compare performance of in-place vs out-of-place DCTs/DSTs
	- Make code strongly typed
		- This is a bit difficult to figure out, as it requires some arithmetic with type variables which isn't possible
		- An alternative is to require that the arrays are `N`-dimensional instead of `N-1`, with a singleton dimension at the end
- Organise
	- Project isn't huge, but still good to split up into different files, with lots of documentation
	- Use multiple dispatch to handle the different boundary conditions and axis types more easily

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
