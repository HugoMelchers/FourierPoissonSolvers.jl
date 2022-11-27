const AnyGrid = Union{Nothing,Offset}

include("periodic.jl")
include("dirichlet.jl")
include("neumann.jl")
include("mixed.jl")

const BoundaryCondition = Union{Periodic,Dirichlet,Neumann}
const AperiodicBC = Union{Dirichlet,Neumann}
const AperiodicBCPair = NTuple{2,AperiodicBC}
const BCPair = Union{NTuple{2,Periodic},AperiodicBCPair}

frequency_response2(ω) = -2 + 2cos(ω)
frequency_response4(ω) = (-30 + 32cos(ω) - 2cos(2ω)) / 12

eigenvalues2(bc1::AperiodicBC, bc2::AperiodicBC, g::AnyGrid, n) = frequency_response2.(wavenumbers(bc1, bc2, g, n))
eigenvalues4(bc1::AperiodicBC, bc2::AperiodicBC, g::AnyGrid, n) = frequency_response4.(wavenumbers(bc1, bc2, g, n))

eigenvalues2(bc1::Periodic, bc2::Periodic, g::AnyGrid, n) = -wavenumbers(bc1, bc2, g, n) .^ 2
eigenvalues4(bc1::Periodic, bc2::Periodic, g::AnyGrid, n) = -wavenumbers(bc1, bc2, g, n) .^ 2
