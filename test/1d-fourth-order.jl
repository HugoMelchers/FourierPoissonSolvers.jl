# @testset "1D-aperiodic-4th-order" begin
#     test_order_of_accuracy_1d_all(x -> sin(2x) / (2 + x), -1, 1, 50:10:100, Val(4))
#     test_order_of_accuracy_1d_all(x -> exp(-x^3), -0.3, 0.7, 20:10:100, Val(4))
#     test_order_of_accuracy_1d_all(x -> log(1.2 + cos(x)), -pi, pi, 20:10:100, Val(4))
# end

@testset "1D-aperiodic-exact-4th-order" begin
    # Second-order accurate methods should also reproduce the exact solution (up to numerical precision) when the exact
    # solution is given by a polynomial of degree 2 or lower.
    f(x) = 1 + x/exp(0.5) - x^2 * pi/3 + x^3 / cos(1) - x^4 * sin(1)/2
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
            grid = (offset,),
            order = Val(4)
        )
        err = solution_error_1d(prob, f, df, d2f)
        @test err < 1e-13
    end
end
