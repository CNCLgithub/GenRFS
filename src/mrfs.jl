export MRFS

struct MRFS{T} <: AbstractRFS{T} end

function Gen.logpdf(r::MRFS{T},
                    xs::AbstractArray{T},
                    elements::RFSElements{T},
                    steps::Int,
                    temp::Float64) where {T}
    !contains(elements, length(xs)) && return -Inf
    @> elements begin
        massociations(xs, steps, temp)
        first
        logsumexp
    end
end
Gen.has_output_grad(::MRFS) = false
Gen.logpdf_grad(::MRFS, value::Vector, args...) = (nothing,)

function Gen.random(::MRFS, elements::RFSElements{T},
                    steps::Int, temp::Float64) where {T}
    sample_elements(elements)
end

(r::MRFS)(es::RFSElements, steps, temp) = Gen.random(r, es, steps, temp)



""" Computes the logscore of every correspondence

Returns a vector where each element is indexed in the partition table.

"""
function massociations(es::RFSElements{T}, xs::Vector{T},
                       steps::Int64, t::Float64) where {T}
    state = RTWState(es, xs)
    for _ = 1:steps
        random_tree_step!(state; t = t)
    end
    n = length(state.logscores_map)
    ls = Vector{Float64}(undef, n)
    pt = Array{Bool}(undef, length(xs), length(es), n)
    @inbounds for (i, (h, l)) in enumerate(state.logscores_map)
        ls[i] = l
        pt[:, :, i] = state.partition_map[h]
    end
    ls, BitArray{3}(pt)
end
