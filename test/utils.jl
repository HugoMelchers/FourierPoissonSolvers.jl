"""
    order_of_accuracy(ns::AbstractArray{Int64}, errors::Vector{Float64})

Given a vector `ns` of positive grid sizes and a vector `errors` of positive errors, compute the order of accuracy by
fitting `log(err) = a - b*log(n)` and returning `b`. In that case `err ≈ exp(a) * n^-b`. so `b` is approximately the
order of accuracy.
"""
function order_of_accuracy(ns::AbstractArray{Int64}, errors::Vector{Float64})
    regression = [ones(Float64, length(ns)) log.(ns)] \ log.(errors)
    -regression[2]
end

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

function compare_solutions_1d(x1, x2, n, f, df, d2f, bc, grid, order)
    if n isa AbstractVector
        return [compare_solutions_1d(x1, x2, k, f, df, d2f, bc, grid, order) for k in n]
    end
    prob = PoissonProblem((n,); boundaries=((bc[1], bc[2]),), lims=(x1, x2), grid=grid, order)
    solution_error_1d(prob, f, df, d2f)
end

function test_order_of_accuracy_1d(f, df, d2f, x1, x2, ns, bc, offset, order::Val{N}) where {N}
    errs = Float64[]
    for n in ns
        prob = PoissonProblem(
            (n,);
            boundaries = ((bc[1], bc[2]),),
            lims = ((x1, x2),),
            grid = (offset,),
            order
        )
        err = solution_error_1d(prob, f, df, d2f)
        push!(errs, err)
    end
    @test isapprox(order_of_accuracy(ns, errs), Float64(N), atol=0.1)
end

function test_order_of_accuracy_1d_all(f, x1, x2, ns, order)
    df(x) = ForwardDiff.derivative(f, x)
    d2f(x) = ForwardDiff.derivative(df, x)
    bc_left_1 = Dirichlet()
    bc_left_2 = Neumann()
    bc_right_1 = Dirichlet()
    bc_right_2 = Neumann()
    for bc_left in (bc_left_1, bc_left_2), bc_right in (bc_right_1, bc_right_2), offset in (nothing, Offset())
        test_order_of_accuracy_1d(f, df, d2f, x1, x2, ns, (bc_left, bc_right), offset, order)
    end
end

