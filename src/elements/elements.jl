
abstract type RandomFiniteElement{T} end

function distribution(::RandomFiniteElement)::Gen.Distribution
    error("not implemented")
end

function args(::RandomFiniteElement)::Tuple
    error("not implemented")
end

function sample_cardinality(::RandomFiniteElement)::Int
    error("not implemented")
end

function sample(rfe::RandomFiniteElement{T})::Vector{T} where {T}
    n = sample_cardinality(rfe)
    sample = Vector{T}(undef, n)
    for i=1:n
        sample[i] = distribution(rfe)(args(rfe)...)
    end
    sample
end

""" The log probability P(|{x}| | rfe)
The log probability of the cardinality of an observation for an RFE.
"""
function cardinality(rfe::RandomFiniteElement, n::Int)
    error("not implemented")
end

"""The probability of P(x | dist(rfe))
The loglikelihood of a component, `x`, given the distribution contained by the element.
"""
function support(rfe::RandomFiniteElement{T}, x::T)::Float64 where T
    Gen.logpdf(distribution(rfe), x, args(rfe)...)
end

abstract type MonomorphicRFE{T} <: RandomFiniteElement{T} end
abstract type EpimorphicRFE{T} <: RandomFiniteElement{T} end
# Intersect
abstract type IsomorphicRFE{T} <: RandomFiniteElement{T} end
# abstract type IsomorphicRFE{T} <: Interesct{MonomorphicRFE{T}, EpimorphicRFE{T}} end


"""The lower and upper bounds of map cardinality"""
function bounds(::RandomFiniteElement)::Tuple{Int64, Int64}
    (0, Inf)
end
bounds(::EpimorphicRFE) = (0, Inf)
bounds(::MonomorphicRFE) = (0, 1)
bounds(::IsomorphicRFE) = (1, 1)

function upper(::RandomFiniteElement)::Int64
    error("Undefined")
end
upper(::EpimorphicRFE) = Inf
upper(::MonomorphicRFE) = 1
upper(::IsomorphicRFE) = 1


function lower(::RandomFiniteElement)::Int64
    error("Undefined")
end
lower(::EpimorphicRFE) = 0
lower(::MonomorphicRFE) = 0
lower(::IsomorphicRFE) = 1


include("bernoulli.jl")
include("poisson.jl")
include("geometric.jl")
