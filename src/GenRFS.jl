module GenRFS

using Gen
using Combinatorics

include("utils.jl")

"""Random Finite Set"""
abstract type RFS{T} <: Gen.Distribution{Vector{T}} end

include("elements/elements.jl")

"""Collection of RFEs"""
const RFSElements{T} = Vector{RandomFiniteElement{T}}


""" Whether the given RFS can support the cardinality of the observation"""
function contains(rfs::RFSElements, n::Int)
    _min, _max = sum.(zip(map(bounds, rfs)...))
    n >= _min && n <= _max
end


""" Generates the partition table for a given set of size `n`.

Only valid when the random finite set contains the observed set.
"""
function partition(es::RFSElements, n::Int)
    # retrieve the power domain for each element
    bs = map(bounds, es)
    upper = collect(Int64, clamp.(map(last, bs), 0, n))
    lower = collect(Int64, clamp.(map(first, bs), 0, n))
    idxs = sortperm(upper, rev = true)
    (idxs, partition_table(upper[idxs], lower[idxs], n))
end

# function associations(es::RFSElements{T}, xs::Vector{T}) where {T}
#     nx = length(xs)
#     images = image.(es)
#     lls = support.(es, extended_xs)
#     partitions = partition(images)
#     lpdfs = Vector{Float64}(undef, length(parts))
#     for (i, part) in enumerate(partitions)
#         lpdf_part = 0
#         for (j, assoc) in enumerate(part)
#             # the last row represent assoc -> []
#             lpdfs_part += cardinality(es[j], assoc)
#             isinf(lpdfs_part) && break
#             lpfds_part += logsumexp(lls[j, assoc])
#         end
#         lpdfs[i] = lpdfs_part
#     end
#     lpdfs
# end

# function Gen.logpdf(rfs::RFS{T}, xs::Vector{T}, elements::RFSElements{T})
#     contains(elements, length(xs)) ? logsumexp(associations(elements, xs)) : -Inf
# end

# function Gen.random(rfs::RFS{T}, elements::Vector{RFE{T}})
#     flatten(sample.(elements))::Vector{T}
# end

end # module
