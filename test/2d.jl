function solution_error_2d(prob, f, fx, fy, Δf)
    _bcs = prob.boundaries
    (xs, ys) = nodes(prob)
    (x1, x2) = prob.lims[1]
    (y1, y2) = prob.lims[2]
    ys = ys'
    vals_x1 = _bcs[1][1] isa Dirichlet ? f.(x1, ys) : fx.(x1, ys)
    vals_x2 = _bcs[1][2] isa Dirichlet ? f.(x2, ys) : fx.(x2, ys)
    vals_y1 = _bcs[2][1] isa Dirichlet ? f.(xs, y1) : fy.(xs, y1)
    vals_y2 = _bcs[2][2] isa Dirichlet ? f.(xs, y2) : fy.(xs, y2)
    rhs = with_boundaries(Δf.(xs, ys), (vals_x1, vals_x2), (vals_y1, vals_y2))
    sol_approx = prob \ rhs
    sol_exact = f.(xs, ys)
    is_singular(prob) && zeromean!(sol_exact)
    maximum(abs, sol_exact - sol_approx)
end

function test_order_of_accuracy_2d(f, x1, x2, y1, y2, ns, bcs, grid)
    fx(x, y) = ForwardDiff.derivative(x′ -> f(x′, y), x)
    fy(x, y) = ForwardDiff.derivative(y′ -> f(x, y′), y)
    fxx(x, y) = ForwardDiff.derivative(x′ -> fx(x′, y), x)
    fyy(x, y) = ForwardDiff.derivative(y′ -> fy(x, y′), y)
    Δf(x, y) = fxx(x, y) + fyy(x, y)
    errs = Float64[]
    for n in ns
        prob = PoissonProblem(
            (n, n);
            boundaries=bcs,
            lims=((x1, x2), (y1, y2)),
            grid
        )
        err = solution_error_2d(prob, f, fx, fy, Δf)
        push!(errs, err)
    end
    @test isapprox(order_of_accuracy(ns, errs), 2.0, atol=0.1)
end

function test_order_of_accuracy_2d_all(f, x1, x2, y1, y2, ns)
    bcs = (Dirichlet(), Neumann())
    grids = (nothing, Offset())
    for (bcx1, bcx2, bcy1, bcy2, xgrid, ygrid) in Iterators.product(bcs, bcs, bcs, bcs, grids, grids)
        test_order_of_accuracy_2d(f, x1, x2, y1, y2, ns, ((bcx1, bcx2), (bcy1, bcy2)), (xgrid, ygrid))
    end
end

@testset "2D-aperiodic" begin
    f(x, y) = (exp(x + 2y) - exp(-2x - y)) * exp(-x^2 - y^2)
    test_order_of_accuracy_2d_all(f, -1, 1, -1, 1, 20:20:200)
end

@testset "2D-aperiodic-exact" begin
    f(x, y) = (x - pi*y)^2 + (x/exp(1) + y)^2 - x*y*cos(1)
    fx(x, y) = ForwardDiff.derivative(x′ -> f(x′, y), x)
    fy(x, y) = ForwardDiff.derivative(y′ -> f(x, y′), y)
    fxx(x, y) = ForwardDiff.derivative(x′ -> fx(x′, y), x)
    fyy(x, y) = ForwardDiff.derivative(y′ -> fy(x, y′), y)
    Δf(x, y) = fxx(x, y) + fyy(x, y)
    nx = 7
    ny = 11
    x1 = -1
    x2 =  1
    y1 = -1
    y2 =  1
    bcs = (Dirichlet(), Neumann())
    for (bc_x1, bc_x2, bc_y1, bc_y2, offset_x, offset_y) in Iterators.product(
        bcs, bcs,
        bcs, bcs,
        (nothing, Offset()),
        (nothing, Offset()),
    )
        bcs_x = (bc_x1, bc_x2)
        bcs_y = (bc_y1, bc_y2)
        prob = PoissonProblem(
            (nx, ny);
            boundaries = (bcs_x, bcs_y),
            grid = (offset_x, offset_y),
            lims = ((x1, x2), (y1, y2))
        )
        err = solution_error_2d(prob, f, fx, fy, Δf)
        if !exact_for_quadratic_solutions(prob)
            @test err > 1e-3
        else
            @test err < 1e-13
        end
    end
end
