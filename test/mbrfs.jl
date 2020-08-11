
# construction
r1 = 0.5
r2 = 0.7
be1 = BernoulliElement{Float64}(r1, normal, (0., 1.0))
be2 = BernoulliElement{Float64}(r2, uniform, (-1, 1.0))
mbrfs = RFSElements{Float64}(undef, 2)
mbrfs[1] = be1
mbrfs[2] = be2

# logpdf
x0 = []
@test logpdf(rfs, x0, mbrfs) == log(1.0 - 1) + log(1.0 - r2)
# TODO: write this out manually
# x1 = [0.]
# @test logpdf(rfs, x1, mbrfs) == log(r) + logpdf(normal, 0, (0.,1.))
# x2 = [-2.0, 0.5]
# @test logpdf(rfs, x2, mbrfs) == -Inf
# x3 = [-2.0, 0.0, 2.0]
