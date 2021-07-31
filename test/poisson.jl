# construction
r1 = 2
p1 = PoissonElement{Float64}(r1, normal, (0., 1.0))
prfs = RFSElements{Float64}(undef, 1)
prfs[1] = p1

# logpdf
x0 = []
@test logpdf(rfs, x0, prfs) == logpdf(poisson, 0, r1)

n = 4
prfs = RFSElements{Float64}(undef, 4)
r = 3
for i = 1:n
    prfs[i] = PoissonElement{Float64}(r, uniform, (-1, 1.0))
end

xs = fill(0.1, 5)
using Profile
using Traceur
using StatProfilerHTML

Profile.init(delay=1.0e-7,
             n = 10^7)
@time logpdf(rfs, xs, prfs);
@time logpdf(rfs, xs, prfs);
# @Traceur.trace(logpdf(rfs, xs, prfs), modules = [GenRFS]);
@profilehtml logpdf(rfs, xs, prfs);
@profilehtml logpdf(rfs, xs, prfs);
