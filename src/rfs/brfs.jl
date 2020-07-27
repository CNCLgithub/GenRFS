# BERNOULLI RFS

export brfs

struct BRFS{T} <: RFS{T} end

struct BRFSParams <: RFSParams
    r::Float64,
    rv::Gen.Distribution
    args::Tuple
end


function partitions(r::BRFS, p::BRFSParams)

end

const brfs = BRFS{Any}()

function Gen.random(::BRFS{T}, params::BRFSParams) where {T}
    r, rv, args = @extract params
    Gen.bernoulli(r) ? [random(rv, args...)] : []
end

function Gen.logpdf(::BRFS{T}, x::Vector{T}, params::BRFSParams)

    r, rv, args = @extract params
    n = length(x)
    lpdf = 0.0
    if n == 0
        lpdf = log(1-r)
    elseif n == 1
        lpdf = log(r) + Gen.logpdf(rv, first(x), args...)
    else
        lpdf = log(0)
    end
    return lpdf
end

(::BRFS)(p::BRFSParams) = Gen.random(BRFS(), p)

Gen.has_output_grad(::BRFS) = false
Gen.logpdf_grad(::BRFS, value::Vector, args...) = (nothing,)

