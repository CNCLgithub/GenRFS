module GenRFS

using Gen
using Combinatorics

"""Random Finite Set"""
abstract type RFS{T} <: Gen.Distribution{Vector{T}} end

include("elements/elements.jl")

"""Collection of RFEs"""
const RFSElements{T} where {T} = Vector{RandomFiniteElement{T}}


# TODO
function partition(images)
    k = length(injective_elements)
    z = n - k + 1
    degrees = clamp.(degrees, 0, n)
    for (e,max_deg) in enumerate(sorted_degrees)
        rem = n - max_deg
        for

    end
end

function associations(es::RFSElements{T}, xs::Vector{T}) where {T}
    nx = length(xs)
    images = image.(es)
    lls = support.(es, extended_xs)
    partitions = partition(images)
    lpdfs = Vector{Float64}(undef, length(parts))
    for (i, part) in enumerate(partitions)
        lpdf_part = 0
        for (j, assoc) in enumerate(part)
            # the last row represent assoc -> []
            lpdfs_part += outer_logpdf(es[j], assoc)
            isinf(lpdfs_part) && break
            lpfds_part +=  logsumexp(lls[j, assoc])
        end
        lpdfs[i] = lpdfs_part
    end
    lpdfs
end

function Gen.logpdf(rfs::RFS{T}, xs::Vector{T}, elements::RFSElements{T})
    logsumexp(associations(elements, xs))
end

function Gen.random(rfs::RFS{T}, elements::Vector{RFE{T}})
    flatten(sample.(elements))::Vector{T}
end

end # module
