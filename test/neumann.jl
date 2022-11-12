function neumann_A(n)
	A = base_A(n)
	A[1, 2] = 2
	A[end, end-1] = 2
	A
end

function neumann_offset_A(n)
	A = base_A(n)
	A[1, 1] = -1
	A[end, end] = -1
	A
end

function poisson_neumann(b)
	n = length(b)
	FFTW.r2r!(b, FFTW.REDFT00)
	l = -2 .+ 2cos.(pi*(0:n-1)./(n-1))
	b ./= l
	b[1] = 0
	FFTW.r2r!(b, FFTW.REDFT00)
	b ./= 2(n-1)
	b
end

function poisson_neumann_offset(b)
	n = length(b)
	FFTW.r2r!(b, FFTW.REDFT10)
	l = -2 .+ 2cos.(pi*(0:n-1)./n)
	b ./= l
	b[1] = 0
	FFTW.r2r!(b, FFTW.REDFT01)
	b ./= 2n
	b
end
