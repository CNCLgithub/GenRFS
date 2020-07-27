
abstract type RandomFiniteElement{T} end

function distribution(::RandomFiniteElement)::Gen.Distribution
    error("not implemented")
end

function args(::RandomFiniteElement)::Tuple
    error("not implemented")
end

""" The log probability P(|{x}| | rfe)
The log probability of the cardinality of an observation for an RFE.
"""
function cardinality(rfe::RandomFiniteElement{T}, x::Vector{T}) where {T}
    error("not implemented")
end

"""The probability of P(x | dist(rfe))
The loglikelihood of a component, `x`, given the distribution contained by the element.
"""
function support(rfe::RandomFiniteElement{T}, x::T) where T
    Gen.logpdf(distribution(d), x, args(rfe))
end

abstract type InjectiveRFE{T} <: RandomFiniteElement{T} end
abstract type SurjectiveRFE{T} <: RandomFiniteElement{T} end
# Intersect
abstract type BijectiveRFE{T} <: RandomFiniteElement{T} end
# abstract type BijectiveRFE{T} <: Interesct{InjectiveRFE{T}, SurjectiveRFE{T}} end

domain(::RandomFiniteElement, k::Int) = error("not implemented")

function domain(::InjectiveRFE, n::Int, d::Int)
    @assert d >=0 && d <= 1
    d == 0 && return repeat([], n)
    collect(0:n)
end
"""Surjective elements have domains from degree [0, 1]"""
function domain(::SurjectiveRFE, n::Int)
    [[],collect(1:n)]
end
domain(::BijectiveRFE, k::Int) = error("please implement me?")
