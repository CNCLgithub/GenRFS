using GenRFS
using Test
using Gen

@testset "RFGM Generative Function" begin
    gm = RFGM(RFS{Float64}(), ())
    be1 = BernoulliElement{Float64}(0.8, normal, (0.0, 1.0))
    be2 = BernoulliElement{Float64}(0.8, uniform, (-1.0, 1.0))
    es = RandomFiniteElement{Float64}[be1, be2]

    # Test simulate
    trace = Gen.simulate(gm, (es,))
    @test trace isa RFSTrace
    @test get_args(trace) == (es,)
    @test isfinite(get_score(trace))

    # Test propose
    choices, weight, xs = Gen.propose(gm, (es,))
    @test isfinite(weight)
    @test choices isa ChoiceMap

    # Test generate
    gen_trace, gen_weight = Gen.generate(gm, (es,), choices)
    @test isapprox(get_score(gen_trace), weight)

    # Test update
    be3 = BernoulliElement{Float64}(0.8, uniform, (-3.0, 3.0))
    new_es = RandomFiniteElement{Float64}[be1, be3]
    argdiffs = (Gen.VectorDiff(2, 2, Dict(2 => UnknownChange())),)
    
    new_trace, weight, retdiff, discard = Gen.update(trace, (new_es,), argdiffs, EmptyChoiceMap())
    @test new_trace isa RFSTrace
    @test isfinite(weight)
end
