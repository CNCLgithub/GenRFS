export MRFS

struct MRFS{T} <: AbstractRFS{T} end

function Gen.logpdf(r::MRFS{T},
                    xs::AbstractArray{T},
                    elements::RFSElements{T},
                    steps::Int,
                    temp::Float64) where {T}
    !contains(elements, length(xs)) && return -Inf
    association_score(r, elements, xs, steps, temp)
end

Gen.has_output_grad(::MRFS) = false
Gen.logpdf_grad(::MRFS, value::Vector, args...) = (nothing,)

function Gen.random(::MRFS, elements::RFSElements{T},
                    steps::Int, temp::Float64) where {T}
    sample_elements(elements)
end

(r::MRFS)(es::RFSElements, steps, temp) = Gen.random(r, es, steps, temp)



"""
   associations(::MRFS, elements, xs, steps, temp)

Computes the logscore of every correspondence
Returns a vector where each element is indexed in the partition table.
"""
function associations(::MRFS{T}, es::RFSElements{T}, xs::AbstractVector{T},
                       steps::Int64, t::Float64) where {T}
    isempty(xs) && return empty_partition_score(es), BitArray{3}([false])

    # Random walk over partition space
    state = RTWState(es, xs)
    for _ = 1:steps
        mcmc_tree_step!(st, t)
    end

    # Extract visited partitions
    n = length(state.visited)
    nx = length(xs)
    ne = length(es)
    
    ls = Vector{Float64}(undef, n)
    pt = zeros(Bool, nx, ne, n) # Initialized to false
    
    @inbounds for (i, (tup_key, l)) in enumerate(state.visited)
        ls[i] = l
        ntuple_to_ptensor!(pt, i, tup_key)
    end
    
    return ls, BitArray{3}(pt)
end


"""

    association_score(::MRFS, elements, xs, steps, temp)::Float64

Integrates across over a random walk of partitions.
Estimation has coverage bias.
"""
function association_score(::MRFS{T},
                           es::RFSElements{T},
                           xs::AbstractVector{T},
                           steps::Int64,
                           t::Float64
                           )::Float64 where {T}
    # Empty observation set -> simply logscore
    isempty(xs) && return empty_partition_score(es)

    # Random walk over partition space
    state = RTWState(es, xs)
    for _ = 1:steps
        mcmc_tree_step!(state, t)
    end
    # log ∑_s exp(score(s)) over the distinct visited partitions
    logsumexp_collection(values(state.visited))
end

"""
    ntuple_to_ptensor!(pt::AbstractArray{Bool, 3}, i::Int, tup_key::NTuple{N, T}) where {N, T}

Populates slice i of 3D tensor `pt` in-place from `tup_key`.
"""
@inline function ntuple_to_ptensor!(pt::AbstractArray{Bool, 3},
                                    i::Int,
                                    tup_key::NTuple{N, T}) where {N, T}
    nx = length(tup_key)
    @inbounds for x in 1:nx
        e = Int(tup_key[x])
        pt[x, e, i] = true
    end
    return nothing
end


function empty_partition_score(es::RFSElements)
    score = -Inf
    for e = es
        score = logsumexp(score, cardinality(e, 0))
    end
    return score
end
