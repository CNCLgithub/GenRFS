
# construction
r = 0.3
be = BernoulliElement{Float64}(r, normal, (0., 1.0))
brfs = RFSElements{Float64}(undef, 1)
brfs[1] = be

# logpdf
x0 = []
@test logpdf(rfs, x0, brfs) == log(1.0 - r)
x1 = [0.]
@test logpdf(rfs, x1, brfs) == log(r) + logpdf(normal, 0, (0.,1.))
x2 = [0., 1.0]
@test logpdf(rfs, x2, brfs) == -Inf
