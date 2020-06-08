# AMORTIZED POISSON MULTI BERNOULLI RFS

export pmbrfs,
        get_td_A,
        PMBRFSParams,
        PMBRFSStats

struct PMBRFS <: Gen.Distribution{Vector} end

struct PMBRFSStats
    partitions::Vector
    assignments::Vector
    ll::Vector
end

mutable struct PMBRFSParams <: RFSParams
    ppp_params::PPPParams
    mbrfs_params::MBRFSParams
    # this is used to hack Gen and get
    # best partitions and assignments
    # and corresponding log likelihoods
    # (user should pass empty PMBRFSStats
    #  when calling random)
    pmbrfs_stats::PMBRFSStats
end

const pmbrfs = PMBRFS()

function Gen.random(::PMBRFS,
                    pmbrfs_params::PMBRFSParams)
    
    ppp_params = pmbrfs_params.ppp_params
    mbrfs_params = pmbrfs_params.mbrfs_params

    # sampling from the poisson part
    p_sample = ppp(ppp_params)
   
    # sampling from the multi bernoulli part
    mb_sample = mbrfs(mbrfs_params)

    return [mb_sample ; p_sample]
end


function Gen.logpdf(::PMBRFS,
                    xs::Vector,
                    pmbrfs_params::PMBRFSParams)

    ppp_params = pmbrfs_params.ppp_params
    mbrfs_params = pmbrfs_params.mbrfs_params

    ppp_masks_lpdfs = Vector{Float64}(undef, length(xs))
    mbrfs_lpdfs = Matrix{Float64}(undef, length(xs), length(mbrfs_params.rs))
   
    # precomputing the individual logpdfs, first loop is over the observed set
    for i=1:length(xs)
        ppp_masks_lpdfs[i] = Gen.logpdf(ppp_params.spatial, xs[i], ppp_params.spatial_params...)
        
        # inner loop over MBRFS components
        for j=1:length(mbrfs_params.rs)
            mbrfs_lpdfs[i,j] = Gen.logpdf(mbrfs_params.rvs[j], xs[i], mbrfs_params.rvs_args[j]...)
        end
    end

    # getting all the partitions
    partitions = []
    for i=0:length(mbrfs_params.rs)
        partitions = [partitions ; collect(combinations(1:length(xs), i))]
    end

    lpdfs_partitions = fill(0.0, length(partitions))
    As = []

    for (i, partition) in enumerate(partitions)
        p_part = setdiff(1:length(xs), partition)
        
        # PPP part
        ppp_lpdf = -ppp_params.rate
        ppp_lpdf += length(p_part)*log(ppp_params.rate)
        for index in p_part
            ppp_lpdf += ppp_masks_lpdfs[index]
        end
        lpdfs_partitions[i] += ppp_lpdf


        # MBRFS part
        # number of components choose number of elements in the sample
        mbrfs_partitions = combinations(1:length(mbrfs_params.rs), length(partition))
        # TODO turn everything into assignments (according to Mario's proposal)
        # assignments = permutations(1:length(mbrfs_params.rs))

        # ldpfs for each tracker existence combination
        mbrfs_lpdfs_partitions = fill(-Inf, length(mbrfs_partitions))
        
        max_As_mbrfs = []
    
        # goes through which components explain the data
        for (j, mbrfs_partition) in enumerate(mbrfs_partitions)

            # figuring out nonexistent components in this particular explanation
            non_existent_comps = setdiff(1:length(mbrfs_params.rs), mbrfs_partition)

            assignments = permutations(mbrfs_partition)
            lpdfs_assignments = zeros(length(assignments))
            
            # goes through data associations
            for (k, assignment) in enumerate(assignments)
                # explaining elements in the sampled set
                # using certain BRFS components of the MBRFS
                for (element, comp) in enumerate(assignment)
                    lpdfs_assignments[k] += log(mbrfs_params.rs[comp]) + mbrfs_lpdfs[partition[element], comp]
                end
                
                # factoring in components not used in the explanation
                # (we assument those BRFSs do not exist)
                for comp in non_existent_comps
                    lpdfs_assignments[k] += log(1.0 - mbrfs_params.rs[comp])
                end
                
                mbrfs_lpdfs_partitions[j] = logsumexp(lpdfs_assignments)
                #println("$k $assignment $(lpdfs_assignments[k])")
            end
            
            # getting the best assignment for this mbrfs component explanation
            push!(max_As_mbrfs, collect(assignments)[argmax(lpdfs_assignments)])
        end
        lpdfs_partitions[i] += logsumexp(mbrfs_lpdfs_partitions)
    
        # getting the best assignment from all mbrfs
        push!(As, max_As_mbrfs[argmax(mbrfs_lpdfs_partitions)])
    end
    
    # saving highest partitions, corresponding assignments
    # and corresponding normalized log likelihoods
    sorted_order = sortperm(lpdfs_partitions, rev=true)
    max_partitions = collect(partitions)[sorted_order][1:3]
    max_assignments = As[sorted_order][1:3]
    _, ll = normalize_weights(lpdfs_partitions[sorted_order][1:3])

    pmbrfs_params.pmbrfs_stats = PMBRFSStats(max_partitions, max_assignments, ll)

    lpdf = logsumexp(lpdfs_partitions)
    
    #println("pmbrfs lpdf: $lpdf")
    return lpdf
end


(::PMBRFS)(pmbrfs_params) = Gen.random(PMBRFS(), pmbrfs_params)

Gen.has_output_grad(::PMBRFS) = false
Gen.logpdf_grad(::PMBRFS, value::Vector, args...) = (nothing,)
