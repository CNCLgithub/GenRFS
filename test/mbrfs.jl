using GenRFS


@gen function sampler(n)
    
    print("asdas")

	rs = [0.3, 0.8, 0.4]
    rvs = fill(normal, 3)
	rvs_args = [(0., 1.), (3., 5.), (-5., 2.)]
    mbrfs_params = MBRFSParams(rs, rvs, rvs_args)	

    for i = 1:n
        sample = @trace(mbrfs(mbrfs_params), :sample => i)
		println(sample)
    end
end

trace, _ = Gen.generate(sampler, (3,))
choices = Gen.get_choices(trace)
println(choices)
