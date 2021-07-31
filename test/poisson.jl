# construction
r1 = 2
p1 = PoissonElement{Float64}(r1, normal, (0., 1.0))
prfs = RFSElements{Float64}(undef, 1)
prfs[1] = p1

# logpdf
x0 = Float64[]
@test logpdf(rfs, x0, prfs) == logpdf(poisson, 0, r1)

n = 4
prfs = RFSElements{Float64}(undef, 4)
r = 3
for i = 1:n
    prfs[i] = PoissonElement{Float64}(r, uniform, (-1, 1.0))
end

xs = fill(0.1, 8)

println("Benchmark: random")
b = @benchmark random(rfs, prfs);
display(b)
println()
println("Benchmark: logpdf")
b = @benchmark logpdf(rfs, xs, prfs);
display(b)
println()
