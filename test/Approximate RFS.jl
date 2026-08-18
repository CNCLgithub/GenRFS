using GenRFS
using Test
using Gen

@testset "Markov RFS (MRFS) Approximation" begin
    rfs_float64 = RFS{Float64}()
    mrfs_float64 = MRFS{Float64}()

    # Construct elements
    es = RandomFiniteElement{Float64}[
        BernoulliElement{Float64}(0.8, normal, (0.0, 0.5)),
        BernoulliElement{Float64}(0.6, normal, (2.0, 0.5)),
        PoissonElement{Float64}(2.0, normal, (4.0, 0.5))
    ]

    xs = [0.1, 2.1, 3.9, 4.2]
    
    # Exact analytic logpdf
    exact_score = Gen.logpdf(rfs_float64, xs, es)

    # Approximate MRFS logpdf
    steps_short = 50
    steps_long = 1000
    temp = 1.0

    approx_short = Gen.logpdf(mrfs_float64, xs, es, steps_short, temp)
    approx_long  = Gen.logpdf(mrfs_float64, xs, es, steps_long, temp)

    @test approx_long > -Inf
    @test isfinite(approx_long)
    @test abs(approx_long - exact_score) <= abs(approx_short - exact_score) + 1e-3
    @test isapprox(approx_long, exact_score, atol = 0.5)
end
