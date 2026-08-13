using GenRFS
using Test
using Gen

@testset "Poisson Multi-Bernoulli RFS" begin

    rfs_float = RFS{Float64}()

    λ1 = 3.0
    p1 = PoissonElement{Float64}(λ1, normal, (0.0, 1.0))
    r2 = 0.4
    b1 = BernoulliElement{Float64}(r2, uniform, (-1.0, 1.0))
    pmbrfs = RandomFiniteElement{Float64}[p1, b1]

    # Empty set x = []
    x0 = Float64[]
    expected_x0 = Gen.logpdf(poisson, 0, λ1) + log(1.0 - r2)
    @test isapprox(Gen.logpdf(rfs_float, x0, pmbrfs), expected_x0)

    # Single observation x = [0.0]
    x1 = [0.0]
    p_p1 = exp(Gen.logpdf(normal, 0.0, 0.0, 1.0))
    p_b1 = exp(Gen.logpdf(uniform, 0.0, -1.0, 1.0))
    
    prob_p1_b0 = exp(Gen.logpdf(poisson, 1, λ1)) * p_p1 * (1.0 - r2)
    prob_p0_b1 = exp(Gen.logpdf(poisson, 0, λ1)) * r2 * p_b1
    
    @test isapprox(Gen.logpdf(rfs_float, x1, pmbrfs), log(prob_p1_b0 + prob_p0_b1))
end
