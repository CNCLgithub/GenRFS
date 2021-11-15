export RFS, rfs

import Base.Iterators: product

struct RFS{T} <: AbstractRFS{T} end

const rfs = RFS{Any}()

function Gen.logpdf(::RFS,
                    xs::AbstractArray,
                    elements::RFSElements{T}) where {T}
    Gen.logpdf(rfs, collect(T, xs), elements)
end
function Gen.logpdf(::RFS,
                    xs::Vector{T},
                    elements::RFSElements{T}) where {T}
    !contains(elements, length(xs)) && return -Inf
    @> elements begin
        associations(xs)
        first
        logsumexp
    end
end
Gen.has_output_grad(::RFS) = false
Gen.logpdf_grad(::RFS, value::Vector, args...) = (nothing,)

function Gen.random(::RFS, elements::RFSElements{T}) where {T}
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
(r::RFS)(es::RFSElements) = Gen.random(r, es)

#################################################################################
# Helpers
#################################################################################

""" Whether the given RFS can support the cardinality of the observation"""
function contains(r::RFSElements, n::Int)::Bool
    _min = 0
    _max = 0
    for e in r
        _min += lower(e)
        _max += upper(e)
    end
    return n >= _min && n <= _max
end

""" Generates the partition table for a given set of size `n`.

Only valid when the random finite set contains the observed set.
"""
function partition(es::RFSElements, s_table::Matrix{Float64})
    ne, nx = size(s_table)
    # no obs
    nx == 0 && return falses(nx, ne, 1)
    # retrieve the size of domain for each element
    us = Vector{Int64}(undef, ne)
    @inbounds for i = 1:length(es)
        us[i] = min(upper(es[i]), nx)
    end
    # compute binary associability table
    a_table = s_table .!== -Inf

    # by pass memoization if cache is set to 0
    if  typeof(partition_ctx.metadata) == LRU{CTX_Key, CTX_Val} &&
        partition_ctx.metadata.maxsize === 0
        return partition_cube(a_table, us)
    end
    mem_partition_cube(a_table, us)
end

function rfs_table(es::RFSElements{T}, xs::AbstractArray,
                   f::Function)::Matrix{Float64} where {T}
    table = Matrix{Float64}(undef, length(es), length(xs))
    for (i,(e,x)) in enumerate(product(es, xs))
        table[i] = f(e,x)
    end
    table
end

""" Computes the logscore of every correspondence

Returns a vector where each element is indexed in the partition table.

"""
function associations(es::RFSElements{T}, xs::Vector{T}) where {T}
    s_table = rfs_table(es, xs, support)
    c_table = rfs_table(es, collect(0:length(xs)), cardinality)
    p_cube = partition(es, s_table)
    nx, ne, np = size(p_cube)
    #no valid partitions found
    ls = np == 0 ? Float64[-Inf] : Vector{Float64}(undef, np)
    @inbounds for p = 1:np
        part_ls = 0.0
        for e in 1:ne
            part_ls === -Inf && continue # no need to continue if -Inf
            _assoc = p_cube[:, e, p]
            nassoc = sum(_assoc)
            part_ls += c_table[e,  nassoc + 1]
            nassoc === 0 && continue # support not valid if empty
            part_ls += nassoc === 1 ? first(s_table[e, _assoc]) : sum(s_table[e, _assoc])
        end
        ls[p] = part_ls
    end
    ls, p_cube
end
