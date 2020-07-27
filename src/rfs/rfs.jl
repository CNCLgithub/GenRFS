function elements(r<:RFS, params<:RFSParams)
   collect(zip(params.rs, params.rvs, params.args))
end

function support(r<:RFS, params<:RFSParams, xs::Vector)
    els = elements(r, params)
    extended = [xs..., []]
    lls = Matrix{Float64}(undef, length(els), length(extended))
    for (i,el) in enumerate(els), (j,x) in enumerate(exended)
        lls[i,j] = Gen.logpdf(first(el), x, tail(el))
    end
    lls
end

function parititions(::RFS, ::RFSParams)
    error("not implemented")
end

function associations(rfs<:RFS, params<:RFSParams, xs::Vector)
    nx = length(xs)
    lls = support(rfs, params, xs)
    parts = partitions(rfs, params)
    lpdfs = Vector{Float64}(undef, length(parts))
    for (i, part) in enumerate(parts)
        lpdf_part = 0
        for (j, assoc) in enumerate(part)
            # the last row represent assoc -> []
            lpfds_path += (j <= nx) ? lls[i, assoc] : lls[i, nx]
        end
        lpdfs[i] = lpdfs_part
    end
    lpdfs
end



include("brfs.jl")
include("mbrfs.jl")
include("ppp.jl")
#include("pmbrfs.jl")
include("pmbrfs_amortized.jl")
