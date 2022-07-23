export PoissonElement

struct PoissonElement{T} <: EpimorphicRFE{T}
    λ::Float64
    d::Gen.Distribution{T}
    args::Tuple
end

distribution(rfe::PoissonElement) = rfe.d
args(rfe::PoissonElement) = rfe.args

# cardinality(rfe::PoissonElement, n::Int) = n < 0 ? -Inf : -(rfe.λ + loggamma(n + 1))
cardinality(rfe::PoissonElement, n::Int) = n < 0 ? -Inf : Gen.logpdf(Gen.poisson, n, rfe.λ)

function sample_cardinality(rfe::PoissonElement)
    Gen.poisson(rfe.λ)
end
