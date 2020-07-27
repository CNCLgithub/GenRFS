struct BernoulliElement{T} <: InjectiveRFE{T}
    r::Float64
    d::Gen.Distribution{T}
    args::Tuple
end

distribution(rfe::BernoulliElement) = rfe.d
args(rfe::BernoulliElement) = rfe.args

function outer_logpdf(b::BernoulliElement, k::Int)
    k > 1 ? -Inf : (k == 1 ? log(r) : log(1.0 - r))
end

function sample(rfe::BernoulliElement)
    rand() > rfe.r ? Gen.random(rfe.d, rfe.args...) : []
end
