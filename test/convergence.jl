using Gen
using Test
using Printf
using GenRFS
using BenchmarkTools

@testset "Analytic (RFS) vs Approximate (MRFS) Benchmark & Accuracy" begin

    rfs_float = RFS{Float64}()
    mrfs_float = MRFS{Float64}()
    
    @testset "Accuracy Convergence" begin
        es = RandomFiniteElement{Float64}[
            BernoulliElement{Float64}(0.7, normal, (-2.0, 0.8)),
            BernoulliElement{Float64}(0.8, normal, (2.0, 0.8)),
            PoissonElement{Float64}(1.5, normal, (0.0, 1.0))
        ]
        
        xs = [-2.1, 1.9, 0.2, 0.1]
        
        exact_val = Gen.logpdf(rfs_float, xs, es)
        
        for steps in [10, 100, 500, 2000]
            approx_val = Gen.logpdf(mrfs_float, xs, es, steps, 1.0)
            @test approx_val <= exact_val + 1e-6
        end
        
        approx_2000 = Gen.logpdf(mrfs_float, xs, es, 2000, 1.0)
        @test isapprox(approx_2000, exact_val, atol=0.2)
    end

    @testset "Performance Scaling Comparison" begin
        function benchmark_scaling(n_obs_list::Vector{Int}, n_elements::Int)
            println("\n" * "="^65)
            println("  ANALYTIC (RFS) vs APPROXIMATE (MRFS) PERFORMANCE BENCHMARK")
            println("="^65)
            @printf("%-10s | %-16s | %-16s | %-12s\n", "Obs (N)", "Analytic (RFS)", "Approx (MRFS)", "Accuracy Δ")
            println("-"^65)
            
            es = RandomFiniteElement{Float64}[
                (i <= n_elements ÷ 2 ?
                    BernoulliElement{Float64}(0.8, normal, (Float64(i), 0.5)) :
                    PoissonElement{Float64}(1.5, normal, (Float64(i), 0.5)))
                for i in 1:n_elements
            ]
            
            for N in n_obs_list
                xs = randn(N)
                
                # Measure Analytic RFS
                Gen.logpdf(rfs_float, xs, es)
                t_exact = @elapsed exact_val = Gen.logpdf(rfs_float, xs, es)
                
                # Measure Approximate MRFS (100 steps)
                Gen.logpdf(mrfs_float, xs, es, 150, 1.0)
                t_approx = @elapsed approx_val = Gen.logpdf(mrfs_float, xs, es, 150, 1.0)
                
                diff = abs(exact_val - approx_val)
                @printf("%-10d | %-14.6f s | %-14.6f s | %-12.4f\n", N, t_exact, t_approx, diff)
                
                @test isfinite(exact_val)
                @test isfinite(approx_val)
            end
            println("="^65 * "\n")
        end

        benchmark_scaling([2, 4, 8, 10], 6)
    end
end
