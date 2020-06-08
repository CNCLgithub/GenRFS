# POISSON MULTI BERNOULLI RFS

export pmbrfs,
        get_td_A

struct PMBRFS <: Gen.Distribution{Vector} end

const pmbrfs = PMBRFS()

function Gen.random(::PMBRFS,
                    ppp_params::PPPParams,
                    mbrfs_params::MBRFSParams)
   
    # sampling from the poisson part
    p_sample = ppp(ppp_params)
   
    # sampling from the multi bernoulli part
    mb_sample = mbrfs(mbrfs_params)

    return [mb_sample ; p_sample]
end



function Gen.logpdf(::PMBRFS,
                    xs::Vector,
                    ppp_params::PPPParams,
                    mbrfs_params::MBRFSParams)
    
    # getting all the partitions
    partitions = []
    for i=0:length(mbrfs_params.rs)
        partitions = [partitions ; collect(combinations(1:length(xs), i))]
    end
    
    lpdfs_partitions = fill(-Inf, length(partitions))

    for (i, partition) in enumerate(partitions)
        p_part = setdiff(1:length(xs), partition)
        lpdfs_partitions[i] = Gen.logpdf(ppp, xs[p_part], ppp_params) + Gen.logpdf(mbrfs, xs[partition], mbrfs_params)
    end
    
    lpdf = logsumexp(lpdfs_partitions)
    
    #println("pmbrfs lpdf: $lpdf")
    return lpdf
end

# hacking gen to get target designation
function get_td_A(::PMBRFS,
                    xs::Vector,
                    ppp_params::PPPParams,
                    mbrfs_params::MBRFSParams)
    # getting all the partitions
    partitions = []
    for i=0:length(mbrfs_params.rs)
        partitions = [partitions ; collect(partitioninations(1:length(xs), i))]
    end
    
    lpdfs_partitions = fill(-Inf, length(partitions))

    As = Vector{Vector{Int}}(undef, length(partitions))

    saved_index = 0
    for (i, partition) in enumerate(partitions)
        p_part = setdiff(1:length(xs), partition)
        lpdfs_partitions[i] = Gen.logpdf(ppp, xs[p_part], ppp_params)
        lpdfs_partitions[i], As[i] = get_A(mbrfs, xs[partition], mbrfs_params)
    end

    # TODO maybe something more interesting
    # and probabilistically proper than max TD of the particle
    sorted_order = sortperm(lpdfs_partitions, rev=true)
    max_tds = collect(partitions)[sorted_order][1:3]
    max_As = As[sorted_order][1:3]
    _, max_lpdfs = normalize_weights(lpdfs_partitions[sorted_order][1:3])

    return max_tds, max_As, max_lpdfs
end

(::PMBRFS)(ppp_params, mbrfs_params) = Gen.random(PMBRFS(), ppp_params, mbrfs_params)

Gen.has_output_grad(::PMBRFS) = false
Gen.logpdf_grad(::PMBRFS, value::Vector, args...) = (nothing,)
