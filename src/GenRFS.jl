module GenRFS

#################################################################################
# Dependencies
#################################################################################

using Gen
using Lazy: @>, @>>


export AbstractRFS

#################################################################################
# Random Finite Element
#################################################################################

include("elements/elements.jl")

#################################################################################
# Random Finite Set
#################################################################################

"""Abstract Random Finite Sets

Defines a distribution over sets implemented as Vector of type `T`
Parameterized by a collection of random finite elements
"""
abstract type AbstractRFS{T} <: Gen.Distribution{Vector{T}} end


# random finite tree search
include("rft.jl")
# random finite sets
include("rfs.jl")

# markov search
include("tree_walk.jl")

end # module
