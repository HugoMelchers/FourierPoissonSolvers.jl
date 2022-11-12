using SpectralPoissonSolvers
using Test
using ForwardDiff

"""
    order_of_accuracy(hs, errors)

Given a vector `hs` of positive step sizes and a vector `errors` of positive errors, compute the order of accuracy by
fitting `log(err) = a + b*log(h)` and returning `b`. In that case `err ≈ exp(a) * h^b`. so `b` is approximately the
order of accuracy.
"""
function order_of_accuracy(hs::Vector{T}, errors::Vector{T}) where T <: Real
    regression = [ones(eltype(hs), length(hs)) log.(hs)] \ log.(errors)
    regression[2]
end

function zeromean!(arr)
    arr .-= sum(arr) / length(arr)
end

function compare_solutions_1d_offset(x1, x2, n, f, d2f, bc)
    if n isa AbstractVector
        return [compare_solutions_1d_offset(x1, x2, k, f, d2f, bc) for k in n]
    end
    xs = range(x1, x2, length=2n+1)[2:2:end]
    xaxis = axis(x2 - x1, n, bc, Offset())
    prob = PoissonProblem((xaxis,), d2f.(xs))
    u1 = f.(xs)
    u2 = solve(prob)
    is_singular(prob) && zeromean!(u1)
    maximum(abs, u1 .- u2)
end

@testset "SpectralPoissonSolvers.jl" begin
    @test begin
        f(x) = exp(sin(x)) - exp(cos(3x))
        df(x) = ForwardDiff.derivative(f, x)
        d2f(x) = ForwardDiff.derivative(df, x)
        ns = 10:10:1000
        bc = Periodic()
        errs = compare_solutions_1d_offset(-pi, pi, ns, f, d2f, Periodic())
        ooa = order_of_accuracy(1 ./ ns, errs)
        isapprox(ooa, 2.0, atol=0.1)
    end
    
    @testset "1D-aperiodic" begin
        f(x) = sin(2x) / (2 + x)
        df(x) = ForwardDiff.derivative(f, x)
        d2f(x) = ForwardDiff.derivative(df, x)
        ns = 10:10:1000
        bc1d = Dirichlet(fill(f(-1), ()))
        bc2d = Dirichlet(fill(f( 1), ()))
        bc1n = Neumann(fill(df(-1), ()))
        bc2n = Neumann(fill(df( 1), ()))

        for bc1 in [bc1d, bc1n], bc2 in [bc2d, bc2n]
            @test begin
                bc = (bc1, bc2)
                errs = compare_solutions_1d_offset(-1, 1, ns, f, d2f, bc)
                ooa = order_of_accuracy(1 ./ ns, errs)
                isapprox(ooa, 2.0, atol = 0.1)
            end
        end
    end
    
    @testset "2D_aperiodic" begin
        f(x, y) = (exp(x + 2y) - exp(-2x - y)) * exp(-x^2 - y^2)
        fx(x, y) = ForwardDiff.derivative(x′ -> f(x′, y), x)
        fy(x, y) = ForwardDiff.derivative(y′ -> f(x, y′), y)
        fxx(x, y) = ForwardDiff.derivative(x′ -> fx(x′, y), x)
        fyy(x, y) = ForwardDiff.derivative(y′ -> fy(x, y′), y)
        
        ns = 10:10:1000
        errs = Float64[]
        for n in ns
            xs = LinRange(-1, 1, 2n+1)[2:2:end]
            rhs = fxx.(xs, xs') .+ fyy.(xs, xs')
            bc1x = Neumann(fx.([-1], xs))
            bc2x = Neumann(fx.([ 1], xs))
            bc1y = Neumann(fy.(xs, [-1]))
            bc2y = Neumann(fy.(xs, [ 1]))
            axis1 = axis(2.0, n, (bc1x, bc2x), Offset())
            axis2 = axis(2.0, n, (bc1y, bc2y), Offset())
            prob = PoissonProblem((axis1, axis2), rhs)
            u1 = f.(xs, xs')
            zeromean!(u1)
            u2 = solve(prob)
            err = maximum(abs, u1 - u2)
            push!(errs, err)
        end
    end
end
