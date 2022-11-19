function solution_error_1d(prob, f, df, d2f)
    _bcs = prob.boundaries
    (xs,) = nodes(prob)
    (x1, x2) = prob.lims[1]
    vals_x1 = _bcs[1][1] isa Dirichlet ? f(x1) : df(x1)
    vals_x2 = _bcs[1][2] isa Dirichlet ? f(x2) : df(x2)
    rhs = with_boundaries(d2f.(xs), (vals_x1, vals_x2))
    sol_approx = prob \ rhs
    sol_exact = f.(xs)
    is_singular(prob) && zeromean!(sol_exact)
    maximum(abs, sol_exact - sol_approx)
end

function compare_solutions_1d(x1, x2, n, f, df, d2f, bc, grid=nothing)
    if n isa AbstractVector
        return [compare_solutions_1d(x1, x2, k, f, df, d2f, bc, grid) for k in n]
    end
    prob = PoissonProblem((n,); boundaries=((bc[1], bc[2]),), lims=((x1, x2),), grid=(grid,))
    solution_error_1d(prob, f, df, d2f)
end

function test_order_of_accuracy_1d(f, df, d2f, x1, x2, ns, bc, offset)
    errs = Float64[]
    for n in ns
        prob = PoissonProblem(
            (n,);
            boundaries = ((bc[1], bc[2]),),
            lims = ((x1, x2),),
            grid = (offset,)
        )
        err = solution_error_1d(prob, f, df, d2f)
        push!(errs, err)
    end
    @test isapprox(order_of_accuracy(ns, errs), 2.0, atol=0.1)
end

function test_order_of_accuracy_1d_all(f, x1, x2, ns)
    df(x) = ForwardDiff.derivative(f, x)
    d2f(x) = ForwardDiff.derivative(df, x)
    bc_left_1 = Dirichlet()
    bc_left_2 = Neumann()
    bc_right_1 = Dirichlet()
    bc_right_2 = Neumann()
    for bc_left in (bc_left_1, bc_left_2), bc_right in (bc_right_1, bc_right_2), offset in (nothing, Offset())
        test_order_of_accuracy_1d(f, df, d2f, x1, x2, ns, (bc_left, bc_right), offset)
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
    errs = compare_solutions_1d(-pi, pi, ns, f, df, d2f, (Periodic(), Periodic()))
    ooa = order_of_accuracy(ns, errs)
    @test isapprox(ooa, 2.0, atol=0.1)
    errs = compare_solutions_1d(-pi, pi, ns, f, df, d2f, (Periodic(), Periodic()), Offset())
    ooa = order_of_accuracy(ns, errs)
    @test isapprox(ooa, 2.0, atol=0.1)
end

@testset "1D-aperiodic" begin
    test_order_of_accuracy_1d_all(x -> sin(2x) / (2 + x), -1, 1, 20:20:200)
    test_order_of_accuracy_1d_all(x -> exp(-x^3), -0.3, 0.7, 20:20:200)
    test_order_of_accuracy_1d_all(x -> log(1.2 + cos(x)), -pi, pi, 20:20:200)
end

@testset "1D-aperiodic-exact" begin
    # Second-order accurate methods should also reproduce the exact solution (up to numerical precision) when the exact
    # solution is given by a polynomial of degree 2 or lower.
    f(x) = 1 + x/exp(0.5) - x^2 * pi/3
    x1 = -1
    x2 = 1
    df(x) = ForwardDiff.derivative(f, x)
    d2f(x) = ForwardDiff.derivative(df, x)
    bc_left_1 = Dirichlet()
    bc_left_2 = Neumann()
    bc_right_1 = Dirichlet()
    bc_right_2 = Neumann()
    n = 7
    for bc_left in (bc_left_1, bc_left_2), bc_right in (bc_right_1, bc_right_2), offset in (nothing, Offset())
        prob = PoissonProblem(
            (n,);
            boundaries = ((bc_left, bc_right),),
            lims = ((x1, x2),),
            grid = (offset,)
        )
        err = solution_error_1d(prob, f, df, d2f)
        if !exact_for_quadratic_solutions(prob)
            @test err > 1e-3
        else
            @test err < 1e-15
        end
    end
end
