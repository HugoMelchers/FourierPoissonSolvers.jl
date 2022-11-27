# TODO: figure out how to properly test the spectrally accurate cases
# @testset "1D-periodic" begin
#     f(x) = exp(sin(x)) - exp(cos(3x))
#     df(x) = ForwardDiff.derivative(f, x)
#     d2f(x) = ForwardDiff.derivative(df, x)
#     ns = 20:20:200
#     bc = Periodic()
#     errs = compare_solutions_1d(-pi, pi, ns, f, df, d2f, (Periodic(), Periodic()))
#     ooa = order_of_accuracy(ns, errs)
#     @test isapprox(ooa, 2.0, atol=0.1)
#     errs = compare_solutions_1d(-pi, pi, ns, f, df, d2f, (Periodic(), Periodic()), Offset())
#     ooa = order_of_accuracy(ns, errs)
#     @test isapprox(ooa, 2.0, atol=0.1)
# end

@testset "1D-aperiodic" begin
    test_order_of_accuracy_1d_all(x -> sin(2x) / (2 + x), -1, 1, 20:20:200, Val(2))
    test_order_of_accuracy_1d_all(x -> exp(-x^3), -0.3, 0.7, 20:20:200, Val(2))
    test_order_of_accuracy_1d_all(x -> log(1.2 + cos(x)), -pi, pi, 20:20:200, Val(2))
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
