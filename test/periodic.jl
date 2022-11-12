function periodic_A(n)
	A = sparse(dirichlet_A(n))
	A[1, n] = A[n, 1] = 1.0
	A
end

function poisson_periodic(b)
	b2 = FFTW.fft(b)
	b2 ./= -2 .+ 2cos.(2pi*FFTW.fftfreq(length(b)))
	b2[1] = 0.0
	real(FFTW.ifft(b2))
end

function poisson_periodic2(b)
	# we can also use a discrete Hartley transform instead of a FFT
	n = length(b)
	b2 = FFTW.r2r(b, FFTW.DHT)
	b2 ./= -2 .+ 2cos.(2pi*FFTW.fftfreq(n))
	b2[1] = 0.0
	FFTW.r2r!(b2, FFTW.DHT)
	b2 ./= n
end
