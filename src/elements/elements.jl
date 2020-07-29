
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

abstract type MonomorphicRFE{T} <: RandomFiniteElement{T} end
abstract type EpimorphicRFE{T} <: RandomFiniteElement{T} end
# Intersect
abstract type IsomorphicRFE{T} <: RandomFiniteElement{T} end
# abstract type IsomorphicRFE{T} <: Interesct{MonomorphicRFE{T}, EpimorphicRFE{T}} end


bounds(::RandomFiniteElement) = (0, Inf)
bounds(::EpimorphicRFE) = (0, Inf)
bounds(::MonomorphicRFE) = (0, 1)
bounds(::IsomorphicRFE) = (1,)


map(::RandomFiniteElement, k::Int) = error("not implemented")

function map(::MonomorphicRFE, n::Int, d::Int)
    @assert d >=0 && d <= 1
    d == 0 && return repeat([], n)
    collect(0:n)
end
"""Epimorphic elements have domains from degree [0, 1]"""
function map(::EpimorphicRFE, n::Int)
    [[],collect(1:n)]
end

""" The map over an isomorphic RFE.

The cardinality, `k == 1`.
"""
function map(::IsomorphicRFE, n::Int, k::Int)
    @assert k == 1
    collect(1:n)
end

include("bernoulli.jl")
include("poisson.jl")
