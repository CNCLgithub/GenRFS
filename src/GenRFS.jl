module GenRFS

using Gen
using Combinatorics

"""Random Finite Set"""
abstract type RFS{T} <: Gen.Distribution{Vector{T}} end

include("elements/elements.jl")

"""Collection of RFEs"""
const RFSElements{T} where {T} = Vector{RandomFiniteElement{T}}


""" Whether the given RFS can support the cardinality of the observation"""
function contains(rfs::RFSElements, n::Int)
    _min, _max = sum.(zip(map(bounds, rfs)...))
    n >= _min && n <= _max
end


function mono_press(m::Int, k::Int)
    a = filter(x -> length(x) <= m, integer_partitions(k))
    table = convert.(Int64, zeros(length(a), k))
    for i = 1:length(a)
        table[i, 1:(k - i + 1)] = a[i]
    end
    table
end


""" Generates the partition table for a given set of size `n`.

Only valid when the random finite set contains the observed set.
"""
function partition(es::RFSElements, n::Int)
    # first count up any isomorphic elements

    n_iso = length(isos)
    iso_table = matrix{Int64}(undef, n, n_iso)
    for i = 1:n_iso
        iso_table[:, i] = circshift(map(isos[i], n), i - 1)
    end
    dof = n - n_iso

    table = []
    # go through epimorphic elements


    # go through monomorphic elements

    n_mono = length(mono)

    for i = 1:n_mono
        others = tail(circshift(mono, i - 1))
    end


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
            lpdfs_part += cardinality(es[j], assoc)
            isinf(lpdfs_part) && break
            lpfds_part += logsumexp(lls[j, assoc])
        end
        lpdfs[i] = lpdfs_part
    end
    lpdfs
end

function Gen.logpdf(rfs::RFS{T}, xs::Vector{T}, elements::RFSElements{T})
    contains(elements, length(xs)) ? logsumexp(associations(elements, xs)) : -Inf
end

function Gen.random(rfs::RFS{T}, elements::Vector{RFE{T}})
    flatten(sample.(elements))::Vector{T}
end

end # module
