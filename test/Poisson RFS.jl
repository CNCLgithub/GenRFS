using Gen
using Test
using GenRFS

@testset "Poisson RFS" begin

    rfs_float = RFS{Float64}()
    
    λ = 2.5
    pe = PoissonElement{Float64}(λ, normal, (0.0, 1.0))
    prfs = RandomFiniteElement{Float64}[pe]

    # Empty set x = []
    x0 = Float64[]
    @test isapprox(Gen.logpdf(rfs_float, x0, prfs), Gen.logpdf(poisson, 0, λ))

    # Single observation x = [0.5]
    x1 = [0.5]
    expected_x1 = Gen.logpdf(poisson, 1, λ) + Gen.logpdf(normal, 0.5, 0.0, 1.0)
    @test isapprox(Gen.logpdf(rfs_float, x1, prfs), expected_x1)

    # Two observations x = [0.0, 1.0]
    x2 = [0.0, 1.0]
    expected_x2 = Gen.logpdf(poisson, 2, λ) +
        Gen.logpdf(normal, 0.0, 0.0, 1.0) +
        Gen.logpdf(normal, 1.0, 0.0, 1.0)
    @test isapprox(Gen.logpdf(rfs_float, x2, prfs), expected_x2)
end
