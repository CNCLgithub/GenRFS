using GenRFS
using Gen
using Test

const gm_bern = RFGM(RFS{Float64}(), ())   # estimator args unused by RFS
const gm_pois = RFGM(MRFS{Float64}(), (1000, 1.0))

function make_trace(gm, es, xs)
    choices = choicemap()
    for (i, x) in enumerate(xs)
        choices[i] = x
    end
    tr, _ = Gen.generate(gm, (es,), choices)
    tr
end

@testset "RFGM generate/simulate" begin
    es = [
        BernoulliElement{Float64}(0.9, normal, (0.0, 0.5)),
        BernoulliElement{Float64}(0.9, normal, (3.0, 0.5)),
        PoissonElement{Float64}(1.5, normal, (6.0, 0.5)),
    ]
    tr = Gen.simulate(gm_bern, (es,))
    @test length(Gen.get_retval(tr)) >= 2          # at least the two isos
    @test Gen.get_score(tr) > -Inf
    # constrained generate must respect the observation
    tr2 = make_trace(gm_bern, es, [0.1, 3.1, 6.0, 6.2])
    @test Gen.get_retval(tr2) ≈ [0.1, 3.1, 6.0, 6.2]
    # impossible observation errors
    # @test_throws ErrorException make_trace(gm_bern, es, [0.1, 0.2, 0.3])
end

@testset "RFGM update: incremental == from scratch" begin
    es0 = [
        BernoulliElement{Float64}(0.9, normal, (0.0, 0.5)),
        BernoulliElement{Float64}(0.9, normal, (3.0, 0.5)),
        PoissonElement{Float64}(1.5, normal, (6.0, 0.5)),
    ]
    xs = [0.1, 3.05, 6.1, 5.9]
    tr = make_trace(gm_bern, es0, xs)

    # move element 3 (Poisson) and leave others fixed
    es1 = [
        BernoulliElement{Float64}(0.9, normal, (0.0, 0.5)),
        BernoulliElement{Float64}(0.9, normal, (3.0, 0.5)),
        PoissonElement{Float64}(1.5, normal, (7.5, 0.5)),   # moved
    ]
    vdiff = Gen.VectorDiff(3, 3, Dict(3 => UnknownChange()))
    new_tr, weight, _, _ = Gen.update(tr, (es1,), (vdiff,), EmptyChoiceMap())

    fresh = make_trace(gm_bern, es1, xs)
    @test isapprox(Gen.get_score(new_tr), Gen.get_score(fresh); atol = 1e-6)
    @test isapprox(weight, Gen.get_score(fresh) - Gen.get_score(tr); atol = 1e-6)
    # visited partition sets should agree as *scored* sets
    for (k, v) in fresh.partitions
        if haskey(new_tr.partitions, k)
            @test isapprox(new_tr.partitions[k], v; atol = 0.5)  # tolerance for walk noise
        end
    end
end

@testset "RFGM regenerate: weight consistency" begin
    es0 = [
        BernoulliElement{Float64}(0.9, normal, (0.0, 0.5)),
        BernoulliElement{Float64}(0.9, normal, (3.0, 0.5)),
        PoissonElement{Float64}(1.5, normal, (6.0, 0.5)),
    ]
    xs = [0.1, 3.05, 6.1]
    tr = make_trace(gm_pois, es0, xs)   # MRFS-backed: exercises association_score path
    es1 = [
        BernoulliElement{Float64}(0.9, normal, (0.0, 0.5)),
        BernoulliElement{Float64}(0.9, normal, (3.0, 0.5)),
        PoissonElement{Float64}(1.5, normal, (6.8, 0.5)),   # jittered
    ]
    vdiff = Gen.VectorDiff(3, 3, Dict(3 => UnknownChange()))
    new_tr, weight, _ = Gen.regenerate(tr, (es1,), (vdiff,), EmptySelection())
    fresh = make_trace(gm_pois, es1, xs)
    @test isapprox(Gen.get_score(new_tr), Gen.get_score(fresh); atol = 0.5)  # walk tolerance
    @test isapprox(weight, Gen.get_score(new_tr) - Gen.get_score(tr); atol = 0.5)
end

@testset "RFGM setdiff update falls back correctly" begin
    es0 = [BernoulliElement{Float64}(0.9, normal, (0.0, 0.5))]
    tr = make_trace(gm_bern, es0, [0.1])
    es1 = [BernoulliElement{Float64}(0.9, normal, (1.0, 0.5)),
           BernoulliElement{Float64}(0.9, normal, (5.0, 0.5))]
    sdiff = Gen.SetDiff(1, 2)
    new_tr, weight, _, _ = Gen.update(tr, (es1,), (sdiff,), EmptyChoiceMap())
    fresh = make_trace(gm_bern, es1, [0.1])
    @test isapprox(Gen.get_score(new_tr), Gen.get_score(fresh); atol = 1e-6)
    @test isapprox(weight, Gen.get_score(fresh) - Gen.get_score(tr); atol = 1e-6)
end
