module GenRFS

using Gen
using Lazy: @>>
using Base.Iterators: take

include("utils.jl")

export AbstractRFS,
    rfs,
    RFSElements,
    AssociationRecord

"""Random Finite Set"""
abstract type AbstractRFS{T} <: Gen.Distribution{AbstractArray{T}} end

struct RFS{T} <: AbstractRFS{T} end

const rfs = RFS{Any}()

include("elements/elements.jl")

const RFSElements{T} = Vector{RandomFiniteElement{T}}

(r::RFS)(es::RFSElements) = Gen.random(r, es)

mutable struct AssociationRecord
    table::PartitionTable
    logscores::Vector{Float64}
    # TODO: better way to initialize record ?
    # At runtime, length of table may be smaller, arrays reassigned
    AssociationRecord(n::Int64) = new(PartitionTable(undef, n),
                                      Vector{Float64}(undef, n))
end

Base.length(xs::AssociationRecord) = length(xs.logscores)

(r::RFS)(es::RFSElements, rec::AssociationRecord) = Gen.random(r, es, rec)

function Gen.logpdf(::RFS, xs, elements::RFSElements{T}) where {T}
    xs = collect(T, xs)
    !contains(elements, length(xs)) && return -Inf
    logsumexp(first(associations(elements, xs)))
end

function Gen.logpdf(::RFS, xs, elements::RFSElements{T}, record::AssociationRecord) where {T}
    xs = collect(T, xs)
    !contains(elements, length(xs)) && return -Inf
    logsumexp(first(associations(elements, xs, record)))
end

function Gen.random(::RFS, elements::RFSElements{T}) where {T}
    collect(T, flatten(sample.(elements)))
end

# TODO: explore ways to amortize association record
function Gen.random(::RFS, elements::RFSElements{T}, record::AssociationRecord) where {T}
    Gen.random(rfs, elements)
end

Gen.has_output_grad(::RFS) = false
Gen.logpdf_grad(::RFS, value::Vector, args...) = (nothing,)

""" Whether the given RFS can support the cardinality of the observation"""
function contains(r::RFSElements, n::Int)
    _min, _max = sum.(zip(map(bounds, r)...))
    n >= _min && n <= _max
end


""" Generates the partition table for a given set of size `n`.

Only valid when the random finite set contains the observed set.
"""
function partition(es::RFSElements, n::Int; fast = true)
    # retrieve the power domain for each element
    bs = map(bounds, es)
    if fast
        upper = collect(Int64, clamp.(map(last, bs), 0, n))
        lower = collect(Int64, clamp.(map(first, bs), 0, n))
    else
        upper = fill(n, length(es))
        lower = fill(0, length(es))
    end
    idxs = sortperm(upper, rev = true)
    # (idxs, partition_table(upper[idxs], lower[idxs], n))
    (idxs, mem_partition_table(upper[idxs], lower[idxs], n))
end

function rfs_table(es::RFSElements{T}, xs, f::Function)::Matrix{Float64} where {T}
    table = Matrix{Float64}(undef, length(es), length(xs))
    for (i,(e,x)) in enumerate(product(es, xs))
        table[i] = f(e,x)
    end
    table
end

""" Computes the logscore of every correspondence

Returns a vector where each element is indexed in the partition table.

"""
function associations(es::RFSElements{T}, xs::Vector{T};
                      fast = true) where {T}
    s_table = rfs_table(es, xs, support)
    c_table = rfs_table(es, collect(0:length(xs)), cardinality)
    p_table = last(partition(es, length(xs), fast = fast))
    ls = Vector{Float64}(undef, length(p_table))
    n_parts = length(es)
    for (i, part) = enumerate(p_table)
        part_ls = 0
        for j in 1:n_parts
            isinf(part_ls) && break # no need to continue if impossible
            assoc = part[j]
            nassoc = length(assoc)
            part_ls += c_table[j, nassoc + 1]
            isempty(assoc) && continue # support not valid if empty
            part_ls += sum(s_table[j, assoc])
        end
        ls[i] = part_ls
    end
    ls, p_table
end

function associations(es::RFSElements{T}, xs::Vector{T},
                      record::AssociationRecord) where {T}
    ls, table = associations(es, xs)
    n = min(length(record), length(table))
    top_n = sortperm(ls, rev = true)[1:n]
    record.table = table[top_n]
    record.logscores = ls[top_n] # last(normalize_weights(ls[top_n]))
    (ls, table)
end


include("analysis/analysis.jl")

end # module
