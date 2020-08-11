export PoissonElement

struct PoissonElement{T} <: EpimorphicRFE{T}
    λ::Int64
    d::Gen.Distribution{T}
    args::Tuple
end

distribution(rfe::PoissonElement) = rfe.d
args(rfe::PoissonElement) = rfe.args

function cardinality(rfe::PoissonElement, n::Int)
    Gen.logpdf(poisson, n, rfe.λ)
end

function sample_cardinality(rfe::PoissonElement)
    Gen.poisson(rfe.λ)
end
