using Gen
using GenRFS
# using Profile
# using StatProfilerHTML
using Random
Random.seed!(1234)

mrfs_float64 = MRFS{Float64}()

# construction
es = RandomFiniteElement{Float64}[]

ne = 4
for i = 1:ne
    mu = rand() * 2.0 * i
    # if rand() > 0.25
    if i > ne
        push!(es, BernoulliElement{Float64}(rand(),
                                               normal,
                                               (mu, 0.5)))
    else
        push!(es, PoissonElement{Float64}(2,
                                             normal,
                                             (mu, 0.5)))
    end

end

nx = 6
xs = randn(nx)

steps = 1000
temp = 10.0
# Gen.logpdf(mrfs_float64, xs, es, steps, temp)
results = convergence(xs, es, steps, temp)
display(results)
