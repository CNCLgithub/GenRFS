using Gen


@gen function sampler(n)
    rv = normal
    rv_args = (0., 1.)
    r = 0.3
    for i = 1:n
        @trace(brfs(r, rv, rv_args), :sample => i)
    end
end

trace, _ = Gen.generate(sampler, (10,))
choices = Gen.get_choices(trace)
println(choices)
