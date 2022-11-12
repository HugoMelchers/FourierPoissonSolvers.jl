function dirichlet_A(n)
	base_A(n)
end

function dirichlet_offset_A(n)
	A = base_A(n)
	A[1, 1] -= 1
	A[end, end] -= 1
	A
end

function poisson_dirichlet(b)
	n = length(b)
	b2 = FFTW.r2r(b, FFTW.RODFT00)
	l = -2 .+ 2cos.(pi*(1:n) ./ (n+1))
	b2 ./= l
	FFTW.r2r!(b2, FFTW.RODFT00)
	b2 ./= 2(n+1)
end

function poisson_dirichlet_offset(b)
	n = length(b)
	b2 = FFTW.r2r(b, FFTW.RODFT10)
	l = -2 .+ 2cos.(pi*(1:n) ./ n)
	b2 ./= l
	FFTW.r2r!(b2, FFTW.RODFT01)
	b2 ./= 2n
end
