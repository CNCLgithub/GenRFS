export MRFS

struct MRFS{T} <: AbstractRFS{T} end

function Gen.logpdf(r::MRFS{T},
                    xs::AbstractArray{T},
                    elements::RFSElements{T},
                    steps::Int,
                    temp::Float64) where {T}
    !contains(elements, length(xs)) && return -Inf
    @> elements begin
        associations(xs, steps, temp)
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
function associations(::MRFS{T}, es::RFSElements{T}, xs::AbstractVector{T},
                       steps::Int64, t::Float64) where {T}
    state = RTWState(es, xs)
    if !isempty(xs)
        for _ = 1:steps
            random_tree_step!(state, t)
        end
    end
    n = length(state.partition_map)
    ls = Vector{Float64}(undef, n)
    pt = Array{Bool, 3}(undef, length(xs), length(es), n)
    @inbounds for (i, (p, l)) in enumerate(state.partition_map)
        ls[i] = l
        pt[:, :, i] = p
    end
    ls, BitArray{3}(pt)
end
