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
            prob = PoissonProblem(
                (n,n);
                boundaries = (Boundary(bc1x, bc2x), Boundary(bc1y, bc2y)),
                lims = ((-1.0,1.0),(-1.0,1.0)),
                grid = (Offset(), Offset())
            )
            u1 = f.(xs, xs')
            zeromean!(u1)
            u2 = prob \ rhs
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
        bcs_x = Boundary(bc_x1, bc_x2)
        bcs_y = Boundary(bc_y1, bc_y2)
        prob = PoissonProblem(
            (nx,ny);
            boundaries = (bcs_x, bcs_y),
            grid = (offset_x, offset_y),
            lims = ((x1, x2),(y1,y2))
        )
        xs = prob.nodes[1]
        ys = prob.nodes[2]'
        prob.boundaries[1].left.values .= bc_x1 isa Dirichlet ? f.(x1, ys) : fx.(x1, ys)
        prob.boundaries[1].right.values .= bc_x2 isa Dirichlet ? f.(x2, ys) : fx.(x2, ys)
        prob.boundaries[2].left.values .= bc_y1 isa Dirichlet ? f.(xs, y1) : fy.(xs, y1)
        prob.boundaries[2].right.values .= bc_y2 isa Dirichlet ? f.(xs, y2) : fy.(xs, y2)
        sol_approx = prob \ (fxx.(xs, ys) .+ fyy.(xs, ys))
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
