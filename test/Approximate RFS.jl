using GenRFS
using Test
using Gen

import GenRFS

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

    @test approx_short > -Inf
    @test approx_long > -Inf
    @test isfinite(approx_long)
    @test abs(approx_long - exact_score) <= abs(approx_short - exact_score) + 1e-3
    @test isapprox(approx_long, exact_score, atol = 0.5)
end

@testset "Analytical dict consistency" begin

    # Construct elements
    es = RandomFiniteElement{Float64}[
        BernoulliElement{Float64}(0.8, normal, (0.0, 0.5)),
        BernoulliElement{Float64}(0.6, normal, (2.0, 0.5)),
        PoissonElement{Float64}(2.0, normal, (4.0, 0.5))
    ]

    xs = [0.1, 2.1, 3.9, 4.2]
    
    visited_exact = GenRFS.associations(es, xs)
    state = GenRFS.RTWState(es, xs)
    for _ = 1:2000; GenRFS.mcmc_tree_step!(state, 1.0); end
    for (key, l) in state.visited
        @test haskey(visited_exact, key)          # walk visited a valid partition
        @test isapprox(visited_exact[key], l; atol = 1e-10)  # scores agree
    end
end

@testset "MRFS vs RFS" begin
    rfs = RFS{Float64}()
    mrfs = MRFS{Float64}()
    temp = 1.0

    @testset "two separated clusters (singles + ensembles)" begin
        es = [
            BernoulliElement{Float64}(0.9, normal, (0.0, 0.2)),
            BernoulliElement{Float64}(0.9, normal, (3.0, 0.2)),
            PoissonElement{Float64}(1.0, normal, (6.0, 0.2)),
        ]
        xs = [0.05, -0.1, 2.95, 3.05, 6.1]
        exact = Gen.logpdf(rfs, xs, es)
        mrfs_score = Gen.logpdf(mrfs, xs, es, 2000, temp)
        @test isapprox(mrfs_score, exact; rtol = 0.1)
    end

    @testset "ambiguous assignment (overlapping elements)" begin
        es = [
            BernoulliElement{Float64}(0.8, normal, (0.0, 0.5)),
            BernoulliElement{Float64}(0.8, normal, (0.3, 0.5)),
            BernoulliElement{Float64}(0.8, normal, (0.6, 0.5)),
        ]
        xs = [0.1, 0.45, 0.7]
        exact = Gen.logpdf(rfs, xs, es)
        for steps = (500, 1000)
            @test isapprox(Gen.logpdf(mrfs, xs, es, steps, temp), exact; atol = 2.0)
        end
        @test abs(Gen.logpdf(mrfs, xs, es, 1000, temp) -
                  Gen.logpdf(mrfs, xs, es, 500, temp)) < 2.0
    end

    @testset "large observation set (nPoisson >> nBernoulli)" begin
        es = [
            PoissonElement{Float64}(3.0, normal, (0.0, 1.0)),
            PoissonElement{Float64}(2.0, normal, (5.0, 1.0)),
            BernoulliElement{Float64}(0.7, normal, (10.0, 0.5)),
        ]
        xs = [collect(-1.5:0.5:1.5); collect(4.0:0.5:6.0); [10.1]]
        exact = Gen.logpdf(rfs, xs, es)
        @test isapprox(Gen.logpdf(mrfs, xs, es, 5000, temp), exact; atol = 1.0)
    end

    @testset "empty and impossible sets" begin
        es = [BernoulliElement{Float64}(0.5, normal, (0.0, 1.0))]
        @test Gen.logpdf(mrfs, Float64[], es, 100, temp) == log(0.5)
        @test Gen.logpdf(mrfs, [0.0, 5.0], es, 100, temp) == -Inf
    end

    @testset "determinism (same seed ⇒ same score)" begin
        es = [
            BernoulliElement{Float64}(0.9, normal, (0.0, 0.2)),
            PoissonElement{Float64}(2.0, normal, (4.0, 0.5)),
        ]
        xs = [0.1, 4.2, 3.8]
        Gen.seed!(42)
        s1 = Gen.logpdf(mrfs, xs, es, 200, temp)
        Gen.seed!(42)
        s2 = Gen.logpdf(mrfs, xs, es, 200, temp)
        @test isapprox(s1, s2)
    end
end
