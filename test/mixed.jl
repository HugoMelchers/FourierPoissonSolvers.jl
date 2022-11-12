function dn_A(n)
	A = base_A(n)
	A[end, end-1] = 2
	A
end

function dn_offset_A(n)
	A = base_A(n)
	A[1, 1] -= 1
	A[end, end] = -1
	A
end

function nd_A(n)
	A = base_A(n)
	A[1, 2] = 2
	A
end

function nd_offset_A(n)
	A = base_A(n)
	A[1, 1] = -1
	A[end, end] -= 1
	A
end

function poisson_dn(b)
	n = length(b)
	FFTW.r2r!(b, FFTW.RODFT01)
	l = -2 .+ 2cos.(pi/n .* ((1:n) .- 0.5))
	b ./= l
	FFTW.r2r!(b, FFTW.RODFT10)
	b./= 2n
	b
end

function poisson_dn_offset(b)
	n = length(b)
	FFTW.r2r!(b, FFTW.RODFT11)
	l = -2 .+ 2cos.(pi/n .* ((1:n) .- 0.5))
	b ./= l
	FFTW.r2r!(b, FFTW.RODFT11)
	b./= 2n
	b
end

function poisson_nd(b)
	n = length(b)
	FFTW.r2r!(b, FFTW.REDFT01)
	l = -2 .+ 2cos.(pi/n .* ((1:n) .- 0.5))
	b ./= l
	FFTW.r2r!(b, FFTW.REDFT10)
	b./= 2n
	b
end

function poisson_nd_offset(b)
	n = length(b)
	FFTW.r2r!(b, FFTW.REDFT11)
	l = -2 .+ 2cos.(pi/n .* ((1:n) .- 0.5))
	b ./= l
	FFTW.r2r!(b, FFTW.REDFT11)
	b./= 2n
	b
end
