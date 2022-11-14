function compare_solutions_1d(x1, x2, n, f, d2f, bc, offset=nothing)
    if n isa AbstractVector
        return [compare_solutions_1d(x1, x2, k, f, d2f, bc, offset) for k in n]
    end
    xaxis = axis(x1, x2, n, bc, offset)
    xs = SpectralPoissonSolvers.xvalues(xaxis)
    prob = PoissonProblem((xaxis,), d2f.(xs))
    u1 = f.(xs)
    u2 = solve(prob)
    is_singular(prob) && zeromean!(u1)
    maximum(abs, u1 .- u2)
end

function test_order_of_accuracy_1d(f, d2f, x1, x2, ns, bc, offset)
    errs = Float64[]
    for n in ns
        ax = axis(x1, x2, n, bc, offset)
        xs = SpectralPoissonSolvers.xvalues(ax)
        rhs = d2f.(xs)
        prob = PoissonProblem((ax,), rhs)
        u_exact = f.(xs)
        u_approx = solve(prob)
        if is_singular(prob)
            zeromean!(u_exact)
        end
        err = maximum(abs, u_exact .- u_approx)
        push!(errs, err)
    end
    @test isapprox(order_of_accuracy(ns, errs), 2.0, atol=0.1)
end

function test_order_of_accuracy_1d_all(f, x1, x2, ns)
    df(x) = ForwardDiff.derivative(f, x)
    d2f(x) = ForwardDiff.derivative(df, x)
    bc_left_1 = Dirichlet(fill(f(x1), (1,)))
    bc_left_2 = Neumann(fill(df(x1), (1,)))
    bc_right_1 = Dirichlet(fill(f(x2), (1,)))
    bc_right_2 = Neumann(fill(df(x2), (1,)))
    for bc_left in (bc_left_1, bc_left_2), bc_right in (bc_right_1, bc_right_2), offset in (nothing, Offset())
        test_order_of_accuracy_1d(f, d2f, x1, x2, ns, (bc_left, bc_right), offset)
    end
end

@testset "1D-periodic" begin
    # TODO: This tests second-order accuracy for the periodic case, but it makes more sense for the code to be
    # opportunistic in choosing to be as accurate as possible, meaning spectral accuracy in the case of periodic
    # boundary conditions.
    f(x) = exp(sin(x)) - exp(cos(3x))
    df(x) = ForwardDiff.derivative(f, x)
    d2f(x) = ForwardDiff.derivative(df, x)
    ns = 20:20:200
    bc = Periodic()
    errs = compare_solutions_1d(-pi, pi, ns, f, d2f, Periodic())
    ooa = order_of_accuracy(ns, errs)
    @test isapprox(ooa, 2.0, atol=0.1)
    errs = compare_solutions_1d(-pi, pi, ns, f, d2f, Periodic(), Offset())
    ooa = order_of_accuracy(ns, errs)
    @test isapprox(ooa, 2.0, atol=0.1)
end

@testset "1D-aperiodic" begin
    f(x) = sin(2x) / (2 + x)
    test_order_of_accuracy_1d_all(f, -1, 1, 20:20:200)
end

@testset "1D-aperiodic-exact" begin
    # Second-order accurate methods should also reproduce the exact solution (up to numerical precision) when the exact
    # solution is given by a polynomial of degree 2 or lower.
    f(x) = 1 + x/exp(0.5) - x^2 * pi/3
    x1 = -1
    x2 = 1
    df(x) = ForwardDiff.derivative(f, x)
    d2f(x) = ForwardDiff.derivative(df, x)
    bc_left_1 = Dirichlet(fill(f(x1), (1,)))
    bc_left_2 = Neumann(fill(df(x1), (1,)))
    bc_right_1 = Dirichlet(fill(f(x2), (1,)))
    bc_right_2 = Neumann(fill(df(x2), (1,)))
    n = 7
    for bc_left in (bc_left_1, bc_left_2), bc_right in (bc_right_1, bc_right_2), offset in (nothing, Offset())
        ax = axis(-1, 1, n, (bc_left, bc_right), offset)
        xs = xvalues(ax)
        prob = PoissonProblem((ax,), d2f.(xs))
        sol = solve(prob)
        sol_exact = f.(xs)
        if is_singular(prob)
            zeromean!(sol_exact)
        end
        @test maximum(abs, sol .- sol_exact) < 1e-15
    end
end
