using Gen
using Test
using GenRFS

@testset verbose=true "GenRFS Tests" begin


    @testset "Bernoulli RFS" begin
        include("Bernoulli RFS.jl")
    end

    @testset "Negative Inf" begin
        include("neg_inf.jl")
    end

    @testset "Poisson RFS" begin
        include("Poisson RFS.jl")
    end

    @testset "Multi-Bernoulli RFS" begin
        include("Multi-Bernoulli RFS.jl")
    end

    @testset "Poisson Multi-Bernoulli RFS" begin
        include("Poisson Multi-Bernoulli.jl")
    end

    @testset "Approximate RFS" begin
        include("Approximate RFS.jl")
    end

    @testset "RFGM Model" begin
        include("RFGM.jl")
        include("RFGM_incremental.jl")
    end

    @testset "Analytic vs Approximate Performance & Accuracy" begin
        include("convergence.jl")
    end

    @testset "Profiling" begin
        include("profile.jl")
    end

end
