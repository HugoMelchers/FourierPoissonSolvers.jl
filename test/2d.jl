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

@testset "2D-aperiodic-exact" begin
    f(x, y) = (x - pi*y)^2 + (x/exp(1) + y)^2 - x*y*cos(1)
    fx(x, y) = ForwardDiff.derivative(x′ -> f(x′, y), x)
    fy(x, y) = ForwardDiff.derivative(y′ -> f(x, y′), y)
    fxx(x, y) = ForwardDiff.derivative(x′ -> fx(x′, y), x)
    fyy(x, y) = ForwardDiff.derivative(y′ -> fy(x, y′), y)
    nx = 7
    ny = 11
    x1 = -1
    x2 =  1
    y1 = -1
    y2 =  1
    bc_x1_d = Dirichlet(zeros(1, ny))
    bc_x2_d = Dirichlet(zeros(1, ny))
    bc_y1_d = Dirichlet(zeros(nx, 1))
    bc_y2_d = Dirichlet(zeros(nx, 1))
    bc_x1_n = Neumann(zeros(1, ny))
    bc_x2_n = Neumann(zeros(1, ny))
    bc_y1_n = Neumann(zeros(nx, 1))
    bc_y2_n = Neumann(zeros(nx, 1))
    for (bc_x1, bc_x2, bc_y1, bc_y2, offset_x, offset_y) in Iterators.product(
        (bc_x1_d, bc_x1_n), (bc_x2_d, bc_x2_n),
        (bc_y1_d, bc_y1_n), (bc_y2_d, bc_y2_n),
        (nothing, Offset()),
        (nothing, Offset()),
    )
        bcs_x = (bc_x1, bc_x2)
        bcs_y = (bc_y1, bc_y2)
        axis_x = axis(x1, x2, nx, bcs_x, offset_x)
        axis_y = axis(y1, y2, ny, bcs_y, offset_y)
        xs = xvalues(axis_x)
        ys = xvalues(axis_y)'
        bc_x1.values .= bc_x1 isa Dirichlet ? f.(x1, ys) : fx.(x1, ys)
        bc_x2.values .= bc_x2 isa Dirichlet ? f.(x2, ys) : fx.(x2, ys)
        bc_y1.values .= bc_y1 isa Dirichlet ? f.(xs, y1) : fy.(xs, y1)
        bc_y2.values .= bc_y2 isa Dirichlet ? f.(xs, y2) : fy.(xs, y2)
        prob = PoissonProblem((axis_x, axis_y), fxx.(xs, ys) .+ fyy.(xs, ys))
        sol_approx = solve(prob)
        sol_exact = f.(xs, ys)
        if is_singular(prob)
            zeromean!(sol_exact)
        end
        if !exact_for_quadratic_solutions(prob)
            @test maximum(abs, sol_approx .- sol_exact) > 1e-3
        else
            @test maximum(abs, sol_approx .- sol_exact) < 1e-13
        end
    end
end
