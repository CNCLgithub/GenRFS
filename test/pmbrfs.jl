r1 = 3
p1 = PoissonElement{Float64}(r1, normal, (0., 1.0))
r2 = 0.4
b1 = BernoulliElement{Float64}(r2, uniform, (-1.0, 1.0))
pmbrfs = RFSElements{Float64}(undef, 2)
pmbrfs[1] = p1
pmbrfs[2] = b1

# logpdf
x0 = []
