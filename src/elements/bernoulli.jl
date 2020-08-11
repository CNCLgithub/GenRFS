export BernoulliElement

struct BernoulliElement{T} <: MonomorphicRFE{T}
    r::Float64
    d::Gen.Distribution{T}
    args::Tuple
end

distribution(rfe::BernoulliElement) = rfe.d
args(rfe::BernoulliElement) = rfe.args

function cardinality(rfe::BernoulliElement, n::Int)
    n > 1 ? -Inf : (n == 1 ? log(rfe.r) : log(1.0 - rfe.r))
end

function sample_cardinality(rfe::BernoulliElement)
    Int64(rand() <= rfe.r)
end
