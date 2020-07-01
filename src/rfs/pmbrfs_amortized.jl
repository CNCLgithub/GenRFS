# AMORTIZED POISSON MULTI BERNOULLI RFS

export pmbrfs,
        get_td_A,
        PMBRFSParams,
        PMBRFSStats

struct PMBRFS <: Gen.Distribution{Vector} end

struct PMBRFSStats
    # partitions
    partitions::Vector
    ll_partitions::Vector
    
    # assignments
    assignments::Vector
    ll_assignments::Vector
end

PMBRFSStats() = PMBRFSStats([],[],[],[])

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
    
    # keeping track of partitions, assignments and their scores
    partitions_dict = Dict()
    assignments_dict = Dict()

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
        # choose which MBRFS components will be explaining the observation
        mbrfs_partitions = combinations(1:length(mbrfs_params.rs), length(partition))

        # ldpfs for each tracker existence combination
        mbrfs_lpdfs_partitions = fill(-Inf, length(mbrfs_partitions))
    
        # collecting the best assignment for each mbrfs_partition
        mbrfs_assignments = Vector{Vector{Int}}(undef, length(mbrfs_partitions))
        
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
                
                # recording assignment and its score (we still need to know the partition)
                assignments_dict[[partition, assignment]] = ppp_lpdf + lpdfs_assignments[k]
            end

            mbrfs_lpdfs_partitions[j] = logsumexp(lpdfs_assignments)
            
            # getting the best assignment for this mbrfs partition explanation
            mbrfs_assignments[j] = collect(assignments)[argmax(lpdfs_assignments)]
        end

        lpdfs_partitions[i] += logsumexp(mbrfs_lpdfs_partitions)
        
        # recording partition and its score (best assignment included)
        best_assignment = mbrfs_assignments[argmax(mbrfs_lpdfs_partitions)]
        partitions_dict[[partition, best_assignment]] = lpdfs_partitions[i]
    end
    
    # sorting partitions and assignments according to their scores
    partitions_pairs = sort(collect(partitions_dict), by=x->x[2], rev=true)
    assignments_pairs = sort(collect(assignments_dict), by=x->x[2], rev=true)
    
    range = 1:3
    pmbrfs_params.pmbrfs_stats = PMBRFSStats([partitions_pairs[i].first for i=range],
                                             # normalize_weights([partitions_pairs[i].second for i=range])[2],
                                             [partitions_pairs[i].second for i=range],
                                             [assignments_pairs[i].first for i=range],
                                             [assignments_pairs[i].second for i=range])

    lpdf = logsumexp(lpdfs_partitions)
    
    return lpdf
end


(::PMBRFS)(pmbrfs_params) = Gen.random(PMBRFS(), pmbrfs_params)

Gen.has_output_grad(::PMBRFS) = false
Gen.logpdf_grad(::PMBRFS, value::Vector, args...) = (nothing,)
