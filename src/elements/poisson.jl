struct PoissonElement{T} <: SurjectiveRFE{T}
    λ::Float64
    d::Gen.Distribution{T}
    args::Tuple
end

distribution(rfe::PoissonElement) = rfe.d
args(rfe::PoissonElement) = rfe.args

function image(rfe::PoissonElement, k::Int)
end

function outer_logpdf(rfe::PoissonElement{T}, x::Vector{T}) where {T}
    Distributions.logpdf(Poisson(rfe.λ), length(x))
end
