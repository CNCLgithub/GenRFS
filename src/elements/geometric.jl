export GeometricElement

struct GeometricElement{T} <: EpimorphicRFE{T}
    p::Float64
    d::Gen.Distribution{T}
    args::Tuple
end

distribution(rfe::GeometricElement) = rfe.d
args(rfe::GeometricElement) = rfe.args


cardinality(rfe::GeometricElement, n::Int) = Gen.logpdf(geometric, n, rfe.p)


function sample_cardinality(rfe::GeometricElement)
    Gen.geometric(rfe.p)
end
