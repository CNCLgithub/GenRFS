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

function softmax(x::Array{Float64}, t::Float64 = 1.0)
    out = similar(x)
    softmax!(out, x, t)
    return out
end

function softmax!(out::Array{Float64}, x::Array{Float64}, t::Float64 = 1.0)
    isempty(x) && return x
    nx = length(x)

    # Single item
    if nx === 1
        out[1] = 1.0
        return nothing
    end

    maxx = maximum(x)

    # Singular mass
    if maxx == Inf
        fill!(out, 0.0)
        out[argmax(x)] = 1.0
        return nothing
    end
        
    # Uniform
    if maxx == -Inf
        out[:] .= 1.0 / nx
        return nothing
    end

    sxs = 0.0
    @inbounds for i = 1:nx
        v = @fastmath exp((x[i] - maxx) / t)
        sxs += v
        out[i] = v
    end
    rmul!(out, 1.0 / sxs)
    return nothing
end

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
