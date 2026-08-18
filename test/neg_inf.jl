using Gen
using Test
using GenRFS

@testset "Negative Inf" begin
    r = 0.3
    be1 = BernoulliElement{Float64}(r, uniform, (0.0, 1.0))
    be2 = BernoulliElement{Float64}(r, uniform, (0.0, 2.0))
    brfs = RandomFiniteElement{Float64}[be1, be2]

    rfs_float = RFS{Float64}()

    # Single observation x = [0.0]
    x1 = [0.7, 1.5]
    @test isfinite(Gen.logpdf(rfs_float, x1, brfs))

end
