export RFS, rfs

import Base.Iterators: product

struct RFS{T} <: AbstractRFS{T} end

const rfs = RFS{Any}()

function Gen.logpdf(rfs::RFS{T},
                    xs::AbstractVector{T},
                    elements::RFSElements{T}) where {T}
    !contains(elements, length(xs)) && return -Inf
    logsumexp(first(associations(rfs, elements, xs)))
end
Gen.has_output_grad(::RFS) = false
Gen.logpdf_grad(::RFS, value::Vector, args...) = (nothing,)

function Gen.random(::RFS{T}, elements::RFSElements{T}) where {T}
    sample_elements(elements)
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

    # # by pass memoization if cache is set to 0
    # if  typeof(partition_ctx.metadata) == LRU{CTX_Key, CTX_Val} &&
    #     partition_ctx.metadata.maxsize === 0
    #     return partition_cube(a_table, us)
    # end
    mem_partition_cube(a_table, us)
end

function rfs_table(es::RFSElements{T}, xs::AbstractArray,
                   f::Function)::Matrix{Float64} where {T}
    table = Matrix{Float64}(undef, length(es), length(xs))
    for (i,(e,x)) in enumerate(product(es, xs))
        @inbounds table[i] = f(e,x)
    end
    table
end

function compute_support_table(es::RFSElements{T}, xs::AbstractVector{T}) where {T}
    ne, nx = length(es), length(xs)
    table = Matrix{Float64}(undef, ne, nx)
    @inbounds for x in 1:nx, e in 1:ne
        table[e, x] = support(es[e], xs[x])
    end
    return table
end

function compute_cardinality_table(es::RFSElements{T}, nx::Int) where {T}
    ne = length(es)
    table = Matrix{Float64}(undef, ne, nx + 1)
    @inbounds for c in 0:nx, e in 1:ne
        table[e, c + 1] = cardinality(es[e], c)
    end
    return table
end

""" Computes the logscore of every correspondence

Returns a vector where each element is indexed in the partition table.

"""
function associations(::RFS{T}, es::RFSElements{T}, xs::AbstractVector{T}) where {T}
    associations(es, xs)
end
function associations(es::RFSElements{T}, xs::AbstractVector{T}) where {T}
    # s_table = rfs_table(es, xs, support)
    # c_table = rfs_table(es, collect(0:length(xs)), cardinality)
    s_table = compute_support_table(es, xs)
    c_table = compute_cardinality_table(es, length(xs))
    p_cube = partition(es, s_table)
    nx, ne, np = size(p_cube)
    #no valid partitions found
    ls = np == 0 ? Float64[-Inf] : Vector{Float64}(undef, np)
    ixs = 1:nx
    ies = 1:ne

    @inbounds for p in 1:np
        part_ls = 0.0
        for e in 1:ne
            nassoc = 0
            for x in 1:nx
                if p_cube[x, e, p]
                    nassoc += 1
                    part_ls += s_table[e, x]
                end
            end
            part_ls += c_table[e, nassoc + 1]
            
            # Correct short-circuit exit on invalid partition
            if part_ls == -Inf
                break
            end
        end
        ls[p] = part_ls
    end

   # @inbounds for p = 1:np
   #      part_ls = 0.0
   #      for e in ies
   #          part_ls === -Inf && continue # no need to continue if -Inf
   #          nassoc = 1
   #          assoc_ls = 0.0
   #          for x = ixs
   #              p_cube[x, e, p] || continue
   #              nassoc += 1
   #              part_ls += s_table[e, x]
   #          end
   #          part_ls += c_table[e, nassoc]
   #      end
   #      ls[p] = part_ls
   #  end
    ls, p_cube
end
