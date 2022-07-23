export RandomFiniteElement, RFSElements

using SpecialFunctions: loggamma

#################################################################################
# Type definitions
#################################################################################
""""A random finite element.

Describes a distrubution over subsets with some cardinality and inner random
 variable.
"""
abstract type RandomFiniteElement{T} end
const RFSElements{T} = Vector{RandomFiniteElement{T}}

#################################################################################
# Methods
#################################################################################

"""The inner random variable of an RFE"""
function distribution(::RandomFiniteElement)::Gen.Distribution
    error("not implemented")
end

"""The parameters of the RFE inner random variable"""
function args(::RandomFiniteElement)::Tuple
    error("not implemented")
end

"""Draw a distribution from the cardinality of the RFE"""
function sample_cardinality(::RandomFiniteElement)::Int
    error("not implemented")
end

function sample_elements(elements::RFSElements{T})::Vector{T} where {T}
    ne = length(elements)
    # number of draws from each element
    ns = Vector{Int64}(undef, ne)
    @inbounds for i = 1:ne
        ns[i] = sample_cardinality(elements[i])
    end
    # populate ranges
    xs = Vector{T}(undef, sum(ns))
    i::Int64 = 1
    @inbounds for j = 1:ne, _ = 1:ns[j]
        e = elements[j]
        xs[i] = distribution(e)(args(e)...)
        i += 1
    end
    xs
end

"""Draw a subset from the rfe"""
function sample_to!(xs::Vector{T}, rfe::RandomFiniteElement{T}) where {T}
    n = sample_cardinality(rfe)
    for _=1:n
        x = distribution(rfe)(args(rfe)...)
        push!(xs, x)
    end
    nothing
end

"""Draw a subset from the rfe"""
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

#################################################################################
# Element map types
#################################################################################

abstract type MonomorphicRFE{T} <: RandomFiniteElement{T} end
abstract type EpimorphicRFE{T} <: RandomFiniteElement{T} end
abstract type IsomorphicRFE{T} <: RandomFiniteElement{T} end

"""The lower and upper bounds of map cardinality"""
function bounds(::RandomFiniteElement)::Tuple{Real, Real}
    (0, Inf)
end
bounds(::EpimorphicRFE) = (0, Inf)
bounds(::MonomorphicRFE) = (0, 1)
bounds(::IsomorphicRFE) = (1, 1)

function upper(::RandomFiniteElement)::Real
    error("Undefined")
end
upper(::EpimorphicRFE) = Inf
upper(::MonomorphicRFE) = 1
upper(::IsomorphicRFE) = 1


function lower(::RandomFiniteElement)::Real
    error("Undefined")
end
lower(::EpimorphicRFE) = 0
lower(::MonomorphicRFE) = 0
lower(::IsomorphicRFE) = 1


#################################################################################
# Concrete Elements
#################################################################################

include("bernoulli.jl")
include("poisson.jl")
include("geometric.jl")
