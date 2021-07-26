module GenRFS

using Gen
using Lazy: @>>
using Base.Iterators:take

include("utils.jl")

export AbstractRFS,
    rfs,
    RFSElements,
    AssociationRecord

"""Random Finite Set"""
abstract type AbstractRFS{T} <: Gen.Distribution{Vector{T}} end

struct RFS{T} <: AbstractRFS{T} end

const rfs = RFS{Any}()

include("elements/elements.jl")

const RFSElements{T} = Vector{RandomFiniteElement{T}}

(r::RFS)(es::RFSElements) = Gen.random(r, es)

mutable struct AssociationRecord
    table
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
    _min = 0
    _max = 0
    for e in r
        l,h = bounds(e)
        _min += l
        _max += h
    end
    n >= _min && n <= _max
end


""" Generates the partition table for a given set of size `n`.

Only valid when the random finite set contains the observed set.
"""
function partition(es::RFSElements, s_table::Matrix{Float64})
    ne, nx = size(s_table)

    # no obs
    nx == 0 && return falses(nx, ne, 1)

    # retrieve the power domain for each element
    upper = Vector{Int64}(undef, ne)
    lower = Vector{Int64}(undef, ne)
    @inbounds for i = 1:length(es)
        _l, _u = bounds(es[i])
        lower[i] = Int64(clamp(_l, 0, nx))
        upper[i] = Int64(clamp(_u, 0, nx))
    end
    # compute binary associability table
    a_table = s_table .> -Inf
    mem_partition_cube(a_table, upper)
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
function associations(es::RFSElements{T}, xs::Vector{T}) where {T}
    @assert !isempty(es) "elements are empty"
    s_table = rfs_table(es, xs, support)
    c_table = rfs_table(es, collect(0:length(xs)), cardinality)
    p_cube = partition(es, s_table)
    nx, ne, np = size(p_cube)
    #no valid partitions found
    ls = np == 0 ? [-Inf] : Vector{Float64}(undef, np)
    @inbounds for p = 1:np
        part_ls = 0
        for e in 1:ne
            isinf(part_ls) && break # no need to continue if impossible
            assoc = p_cube[:, e, p]
            nassoc = sum(assoc)
            part_ls += c_table[e,  nassoc + 1]
            nassoc == 0 && continue # support not valid if empty
            part_ls += sum(s_table[e, assoc])
        end
        ls[p] = part_ls
    end
    ls, p_cube
end

function associations(es::RFSElements{T}, xs::Vector{T},
                      record::AssociationRecord) where {T}
    ls, table = associations(es, xs)
    noninf = findall(!isinf, ls)
    ls = ls[noninf]
    table = table[noninf]
    n = min(length(record), length(noninf))
    top_n = sortperm(ls, rev = true)[1:n]
    record.table = table[top_n]
    record.logscores = ls[top_n] # last(normalize_weights(ls[top_n]))
    (ls, table)
end

end # module
