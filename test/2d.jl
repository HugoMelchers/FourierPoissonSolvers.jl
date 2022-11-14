@testset "2D-aperiodic" begin
    f(x, y) = (exp(x + 2y) - exp(-2x - y)) * exp(-x^2 - y^2)
    fx(x, y) = ForwardDiff.derivative(x′ -> f(x′, y), x)
    fy(x, y) = ForwardDiff.derivative(y′ -> f(x, y′), y)
    fxx(x, y) = ForwardDiff.derivative(x′ -> fx(x′, y), x)
    fyy(x, y) = ForwardDiff.derivative(y′ -> fy(x, y′), y)
    ns = 20:20:200

    @test begin
        errs = Float64[]
        for n in ns
            xs = LinRange(-1, 1, 2n+1)[2:2:end]
            rhs = fxx.(xs, xs') .+ fyy.(xs, xs')
            bc1x = Neumann(fx.([-1], xs'))
            bc2x = Neumann(fx.([ 1], xs'))
            bc1y = Neumann(fy.(xs, [-1]'))
            bc2y = Neumann(fy.(xs, [ 1]'))
            axis1 = axis(-1, 1, n, (bc1x, bc2x), Offset())
            axis2 = axis(-1, 1, n, (bc1y, bc2y), Offset())
            prob = PoissonProblem((axis1, axis2), rhs)
            u1 = f.(xs, xs')
            zeromean!(u1)
            u2 = solve(prob)
            err = maximum(abs, u1 - u2)
            push!(errs, err)
        end
        isapprox(order_of_accuracy(ns, errs), 2.0, atol=0.1)
    end
end
