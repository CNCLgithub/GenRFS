# MULTI BERNOULLI RFS

export mbrfs,
        MBRFSParams,
        get_A

abstract type RFS <: Gen.Distribution{Vector} end

struct MBRFS <: RFS end

struct MBRFSParams <: RFSParams
    rs::Vector{Float64}
    rvs::Vector{Gen.Distribution}
    rvs_args::Vector
end

const mbrfs = MBRFS()

function Gen.random(::MBRFS,
                    params::MBRFSParams)

    sample = []
    
    for i=1:length(params.rs)
        b = brfs(params.rs[i], params.rvs[i], params.rvs_args[i])
        if b != []
            push!(sample, b[1])
        end
    end

	return sample
end

function Gen.logpdf(::MBRFS,
                    xs::Vector,
                    params::MBRFSParams)
    
	# MBRFS can only support sets <= number of components
	if length(xs) > length(params.rs)
		return log(0)
	end
   

	# number of components choose number of elements in the sample
	combs = combinations(1:length(params.rs), length(xs))

    # ldpfs for each tracker existence combination
    lpdfs_combs = fill(-Inf, length(combs))
    
    # goes through which components explain the data
    for (i, comb) in enumerate(combs)

		# figuring out nonexistent components in this particular explanation
		non_existent_comps = setdiff(1:length(params.rs), comb)

		perms = permutations(comb)
        lpdfs_perms = zeros(length(perms))
        
        # goes through data associations
        for (j, perm) in enumerate(perms)
			# explaining elements in the sampled set
			# using certain BRFS components of the MBRFS
			for (element, comp) in enumerate(perm)
                lpdfs_perms[j] += Gen.logpdf(brfs, [xs[element]], params.rs[comp], params.rvs[comp], params.rvs_args[comp])
			end
			
			# factoring in components not used in the explanation
			# (we assument those BRFSs do not exist)
			for comp in non_existent_comps
                lpdfs_perms[j] += Gen.logpdf(brfs, [], params.rs[comp], params.rvs[comp], params.rvs_args[comp])
			end

            lpdfs_combs[i] = logsumexp(lpdfs_perms)
		end
	end

	
	lpdf = logsumexp(lpdfs_combs)
    
    #println("mbrfs pdf: $lpdf")
	return lpdf
end

function get_A(::MBRFS,
               xs::Vector,
               params::MBRFSParams)
    
	# MBRFS can only support sets <= number of components
	if length(xs) > length(params.rs)
        return -Inf, []
	end
   
	# number of components choose number of elements in the sample
	combs = combinations(1:length(params.rs), length(xs))

    # ldpfs for each tracker existence combination
    lpdfs_combs = fill(-Inf, length(combs))
    
    # getting the maximum likelihood A (perm)
    max_A = []
    max_lpdf_A = -Inf
    
    # goes through which components explain the data
    for (i, comb) in enumerate(combs)

		# figuring out nonexistent components in this particular explanation
		non_existent_comps = setdiff(1:length(params.rs), comb)

		perms = permutations(comb)
        lpdfs_perms = zeros(length(perms))
        
        # goes through data associations
        for (j, perm) in enumerate(perms)
			# explaining elements in the sampled set
			# using certain BRFS components of the MBRFS
			for (element, comp) in enumerate(perm)
                lpdfs_perms[j] += Gen.logpdf(brfs, [xs[element]], params.rs[comp], params.rvs[comp], params.rvs_args[comp])
			end
			
			# factoring in components not used in the explanation
			# (we assument those BRFSs do not exist)
			for comp in non_existent_comps
                lpdfs_perms[j] += Gen.logpdf(brfs, [], params.rs[comp], params.rvs[comp], params.rvs_args[comp])
			end

            lpdfs_combs[i] = logsumexp(lpdfs_perms)
		end
        if maximum(lpdfs_perms) > max_lpdf_A
            index = argmax(lpdfs_perms)
            max_lpdf_A = lpdfs_perms[index]
            max_A = collect(perms)[index]
        end
	end

	lpdf = logsumexp(lpdfs_combs)

	return lpdf, max_A
end

(::MBRFS)(params) = Gen.random(MBRFS(), params)

Gen.has_output_grad(::MBRFS) = false
Gen.logpdf_grad(::MBRFS, value::Vector, args...) = (nothing,)
