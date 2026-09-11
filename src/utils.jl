"""
Samples from a categorical distribution with no memory allocation

Note! Does not check if `sum(ws) == 1`
"""
function unsafe_categorical(ws::Array{Float64})
    n = length(ws)
    x = 1
    w = 0.0
    a = rand()
    @inbounds for i = 1:(n-1)
        w += ws[i]
        a < w && break
        x += 1
    end
    return x
end

"""
Fused in-place categorical sampling from unnormalized weights.
Mutates `ws` (normalized in place, temperature t applied). No allocation.
Entries may be 0.0 (self-entry / same-element) or exp(±Inf) = (0.0, Inf) handled via max-subtraction.
Returns the sampled index in 1:length(ws).
"""
function unsafe_categorical!(ws::Vector{Float64}, t::Float64 = 1.0)::Int
    n = length(ws)
    # 1. max-subtract (stability under t; deltas can be large)
    m = -Inf
    @inbounds for v in ws; v > m && (m = v); end
    m == -Inf && return 0                # all entries -Inf: no legal move (shouldn't happen; self=0.0 exists)
    s = 0.0
    @inbounds for i in 1:n
        ws[i] = @fastmath exp((ws[i] - m) / t)
        s += ws[i]
    end
    # 2. normalize in place
    inv_s = 1.0 / s
    @inbounds for i in 1:n
        ws[i] *= inv_s
    end
    # 3. sample — reuse the existing unsafe_categorical logic
    return unsafe_categorical(ws)
end


function softmax(x::Array{Float64}, t::Float64 = 1.0)
    out = similar(x)
    softmax!(out, x, t)
    return out
end

function softmax!(out, x, t)
    m = -Inf
    @inbounds for v in x; v > m && (m = v); end
    s = 0.0
    @inbounds for i in eachindex(x)
        e = @fastmath exp((x[i] - m) / t)
        out[i] = e; s += e
    end
    @inbounds for i in eachindex(x); out[i] /= s; end
    nothing
end
# function softmax!(out::Array{Float64}, x::Array{Float64}, t::Float64 = 1.0,
#                   maxx::Float64=maximum(x))
#     isempty(x) && return x
#     nx = length(x)

#     # Single item
#     if nx === 1
#         out[1] = 1.0
#         return nothing
#     end

#     maxx = maximum(x)

#     # Singular mass
#     if maxx == Inf
#         fill!(out, 0.0)
#         out[argmax(x)] = 1.0
#         return nothing
#     end
        
#     # Uniform
#     if maxx == -Inf
#         out[:] .= 1.0 / nx
#         return nothing
#     end

#     sxs = 0.0
#     @inbounds for i = 1:nx
#         v = @fastmath exp((x[i] - maxx) / t)
#         sxs += v
#         out[i] = v
#     end
#     rmul!(out, 1.0 / sxs)
#     return nothing
# end

function unsafe_find_true(subarray)
    findfirst(subarray)
    # n = length(subarray)
    # x = 0
    # @inbounds for i = 1:n
    #     if subarray[i]
    #         x = i
    #         break
    #     end
    # end
    # return x
end

@inline function logsumexp_collection(vals)
    isempty(vals) && return -Inf
    m = -Inf
    for v in vals; v > m && (m = v); end
    acc = 0.0
    for v in vals; acc += exp(v - m); end
    return m + log(acc)
end
