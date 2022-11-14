"""
    order_of_accuracy(hs, errors)

Given a vector `hs` of positive step sizes and a vector `errors` of positive errors, compute the order of accuracy by
fitting `log(err) = a + b*log(h)` and returning `b`. In that case `err ≈ exp(a) * h^b`. so `b` is approximately the
order of accuracy.
"""
function order_of_accuracy(ns::AbstractArray{Int64}, errors::Vector{Float64})
    regression = [ones(Float64, length(ns)) log.(ns)] \ log.(errors)
    -regression[2]
end
