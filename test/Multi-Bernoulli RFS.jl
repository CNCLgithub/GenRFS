using Gen
using Test
using GenRFS

@testset "Multi-Bernoulli RFS" begin

    rfs_float = RFS{Float64}()

    r1 = 0.5
    r2 = 0.7
    be1 = BernoulliElement{Float64}(r1, normal, (0.0, 1.0))
    be2 = BernoulliElement{Float64}(r2, uniform, (-1.0, 1.0))
    mbrfs = RandomFiniteElement{Float64}[be1, be2]

    # Empty set x = []
    x0 = Float64[]
    expected_x0 = log(1.0 - r1) + log(1.0 - r2)
    @test isapprox(Gen.logpdf(rfs_float, x0, mbrfs), expected_x0)

    # Single observation x = [0.0]
    x1 = [0.0]
    p_be1 = exp(Gen.logpdf(normal, 0.0, 0.0, 1.0))
    p_be2 = exp(Gen.logpdf(uniform, 0.0, -1.0, 1.0))
    prob_x1 = r1 * (1.0 - r2) * p_be1 + (1.0 - r1) * r2 * p_be2
    @test isapprox(Gen.logpdf(rfs_float, x1, mbrfs), log(prob_x1))

    # Two observations x = [-0.5, 0.5]
    x2 = [-0.5, 0.5]
    p11 = exp(Gen.logpdf(normal, -0.5, 0.0, 1.0))
    p12 = exp(Gen.logpdf(uniform, -0.5, -1.0, 1.0))
    p21 = exp(Gen.logpdf(normal, 0.5, 0.0, 1.0))
    p22 = exp(Gen.logpdf(uniform, 0.5, -1.0, 1.0))
    
    prob_x2 = r1 * r2 * (p11 * p22 + p12 * p21)
    @test isapprox(Gen.logpdf(rfs_float, x2, mbrfs), log(prob_x2))

    # Three observations for 2 Bernoulli elements must be -Inf
    x3 = [-0.5, 0.0, 0.5]
    @test Gen.logpdf(rfs_float, x3, mbrfs) == -Inf
end
