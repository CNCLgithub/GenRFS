struct PointMassElement{T} <: BijectiveRFE{T}
    d::Gen.Distribution{T}
    args::Tuple
end

distribution(rfe::PointMassElement) = rfe.d
args(rfe::PointMassElement) = rfe.args

outer_logpdf(rfe::PointMassElement, n::Int) = k == 1 ? 0 : -Inf

sample(rfe::PointMassElement) = Gen.random(rfe.d, args.args...)
