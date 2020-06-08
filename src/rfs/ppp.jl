# POISSON POINT PROCESS
export ppp,
        PPPParams

struct PPP <: Gen.Distribution{Vector} end

struct PPPParams <: RFSParams
    rate::Float64
    spatial::Gen.Distribution
    spatial_params
end

const ppp = PPP()

function Gen.random(::PPP,
                    params::PPPParams)
     
    num = poisson(params.rate)
    sample = []
    for i=1:num
        push!(sample, params.spatial(params.spatial_params...))
    end

    return sample
end

function Gen.logpdf(::PPP,
                    xs::Vector,
                    params::PPPParams)
  
    # PPP pdf is e^(-rate)*product(intensities(xs))
    lpdf = -params.rate

    lpdf += length(xs)*log(params.rate)

    for x in xs
        lpdf += Gen.logpdf(params.spatial, x, params.spatial_params...)
    end
    
    #println("PPP lpdf: $lpdf")
    return lpdf
end

(::PPP)(params) = Gen.random(PPP(), params)

Gen.has_output_grad(::PPP) = false
Gen.logpdf_grad(::PPP, value::Vector, args...) = (nothing,)
