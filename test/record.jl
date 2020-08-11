using Gen

r1 = 3
p1 = PoissonElement{Float64}(r1, normal, (0., 1.0))
r2 = 0.4
b1 = BernoulliElement{Float64}(r2, uniform, (-1.0, 1.0))
pmbrfs = RFSElements{Float64}(undef, 2)
pmbrfs[1] = p1
pmbrfs[2] = b1

record = AssociationRecord(10)
s = rfs(pmbrfs, record)
display(s)
logpdf(rfs, s, pmbrfs)
@time logpdf(rfs, s, pmbrfs)
logpdf(rfs, s, pmbrfs, record)
@time logpdf(rfs, s, pmbrfs, record)

display(collect(zip(record.table, record.logscores)))
