using Gen
using Test
using GenRFS

@testset "Bernoulli RFS" begin
    r = 0.3
    be = BernoulliElement{Float64}(r, normal, (0.0, 1.0))
    brfs = RandomFiniteElement{Float64}[be]

    rfs_float = RFS{Float64}()

    # Empty set x = []
    x0 = Float64[]
    @test isapprox(Gen.logpdf(rfs_float, x0, brfs), log(1.0 - r))

    # Single observation x = [0.0]
    x1 = [0.0]
    expected_x1 = log(r) + Gen.logpdf(normal, 0.0, 0.0, 1.0)
    @test isapprox(Gen.logpdf(rfs_float, x1, brfs), expected_x1)

    # Cardinality > 1 for single Bernoulli element must be -Inf
    x2 = [0.0, 1.0]
    @test Gen.logpdf(rfs_float, x2, brfs) == -Inf
end
