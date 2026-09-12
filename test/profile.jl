using Gen
using Test
using GenRFS
using StatProfilerHTML



@testset "Profiling" begin
    mrfs_float = MRFS{Float64}()
    xs = randn(64)
    n_elements = 50
    nsteps = 100000
    es = RandomFiniteElement{Float64}[
        (i <= n_elements ÷ 2 ?
            BernoulliElement{Float64}(0.8, normal, (Float64(i), 0.5)) :
            PoissonElement{Float64}(1.5, normal, (Float64(i), 0.5)))
        for i in 1:n_elements
            ]

    Gen.logpdf(mrfs_float, xs, es, 150, 1.0)
    @profilehtml Gen.logpdf(mrfs_float, xs, es, nsteps, 1.0)

    @test true
end

