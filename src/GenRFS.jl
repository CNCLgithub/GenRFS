module GenRFS

using Gen

include("utils.jl")

"""Random Finite Set"""
abstract type RFS{T} <: Gen.Distribution{Vector{T}} end

const rfs = RFS{Any}()


"""Collection of RFEs"""
include("elements/elements.jl")

const RFSElements{T} = Vector{RandomFiniteElement{T}}

(::RFS{T})(es::RFSElements{T}) = Gen.random(RFS{T}(), es)


function Gen.logpdf(::RFS{T}, xs::Vector{T}, elements::RFSElements{T})
    !contains(elements, length(xs)) && return -Inf
    logsumexp(associations(elements, xs))
end

function Gen.random(::RFS{T}, elements::RFSElements{T})
    collect(T, flatten(sample.(elements)))
end

Gen.has_output_grad(::RFS) = false
Gen.logpdf_grad(::RFS{T}, value::Vector{T}, args...) = (nothing,)

""" Whether the given RFS can support the cardinality of the observation"""
function contains(r::RFSElements, n::Int)
    _min, _max = sum.(zip(map(bounds, r)...))
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
    (idxs, mem_parititon_table(upper[idxs], lower[idxs], n))
end

function support_table(es::RFSElements{T}, xs::Vector{T}) where {T}
    idx_t = collect(product(es, xs))
    table = Matrix{Float64}(undef, size(idx_t)...)
    for i in eachindex(idx_t)
        table[i] = support(idx_t[i]...)
    end
    table
end

""" Computes the logscore of every correspondence

Returns a vector where each element is indexed in the partition table.
"""
function associations(es::RFSElements{T}, xs::Vector{T}) where {T}
    s_table = support_table(es, xs)
    p_table = last(partition(es, length(xs))
    lpdfs = Vector{Float64}(undef, size(p_table, 1))
    for (i, part) in enumerate(p_table)
        lpdf_part = 0
        for (j, assoc) in enumerate(part)
            lpdfs_part += cardinality(es[j], length(assoc))
            isinf(lpdfs_part) && return [-Inf] # no need to continue if impossible
            isempty(assoc) && continue # support no valid if empty
            lpfds_part += sum(s_table[j, assoc])
        end
        lpdfs[i] = lpdfs_part
    end
    lpdfs
end


end # module
