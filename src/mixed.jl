#=
For an axis with Dirichlet boundary conditions on one side and Neumann on the other, we use a transform that considers
waves that respect those boundary conditions. Note that the different types of transforms are named based on the kind
of symmetry they assume at the left boundary, so the cases with Dirichlet boundary conditions on the left are all called
Discrete Sine Transforms, independently of what the boundary conditions are on the right.
=#

fwd_transform(::Dirichlet, ::Neumann, ::Nothing) = RODFT01
bwd_transform(::Dirichlet, ::Neumann, ::Nothing) = RODFT10

fwd_transform(::Dirichlet, ::Neumann, ::Offset) = RODFT11
bwd_transform(::Dirichlet, ::Neumann, ::Offset) = RODFT11

scalingfactor(::Dirichlet, ::Neumann, ::Union{Nothing, Offset}, n) = 2n
frequencies(::Dirichlet, ::Neumann, ::Union{Nothing, Offset}, n) = LinRange(0, π, 2n+1)[2:2:end]



fwd_transform(::Neumann, ::Dirichlet, ::Nothing) = REDFT01
bwd_transform(::Neumann, ::Dirichlet, ::Nothing) = REDFT10

fwd_transform(::Neumann, ::Dirichlet, ::Offset) = REDFT11
bwd_transform(::Neumann, ::Dirichlet, ::Offset) = REDFT11

scalingfactor(::Neumann, ::Dirichlet, ::Union{Nothing, Offset}, n) = 2n
frequencies(::Neumann, ::Dirichlet, ::Union{Nothing, Offset}, n) = LinRange(0, π, 2n+1)[2:2:end]
