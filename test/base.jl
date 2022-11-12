using LinearAlgebra, SparseArrays
import FFTW

"""
Compute the root-mean-square of a residual vector.
"""
rmse(r) = sqrt(sum(abs2, r) ./ length(r))

"""
Creates the n×n tridiagonal matrix that discretises the Laplace operator (with Dirichlet boundary conditions).
Other boundary conditions are created by changing some entries in this matrix.
"""
function base_A(n)
	Tridiagonal(fill(1.0, n-1), fill(-2.0, n), fill(1.0, n-1))
end

function test_matrix_vs_spectral(A, invA, m, normalise)
	n = size(A, 1)
	l = Float64[]
	for i in 1:m
		b = randn(n)
		x = invA(copy(b))
		r = A * x .- b
		if normalise
			r .-= sum(r) / length(r)
		end
		push!(l, rmse(r))
	end
	l
end

