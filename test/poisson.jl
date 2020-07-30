# construction
r1 = 2
p1 = PoissonElement{Float64}(r1, normal, (0., 1.0))
prfs = RFSElements{Float64}(undef, 1)
prfs[1] = p1

# logpdf
x0 = []
@test logpdf(rfs, x0, prfs) == logpdf(poisson, 0, r1)

prfs = RFSElements{Float64}(undef, 2)
r2 = 4
p2 = PoissonElement{Float64}(r2, uniform, (-1, 1.0))
prfs[1] = p1
prfs[2] = p2
