using Gen
using GenRFS
using GenRFS: rfs_table
using BenchmarkTools
using Profile
using StatProfilerHTML
using UnicodePlots
using Random
Random.seed!(1234)


# construction
mbrfs = RandomFiniteElement{Float64}[]

ne = 4
for i = 1:ne
    mu = rand() * 1.5
    # if rand() > 0.25
    if i > ne
        push!(mbrfs, BernoulliElement{Float64}(rand(),
                                               normal,
                                               (mu, 0.5)))
    else
        push!(mbrfs, PoissonElement{Float64}(2,
                                             normal,
                                             (mu, 0.5)))
    end

end

nx = 6
xs = randn(nx)


# ml = rfs_table(mbrfs, xs, GenRFS.support)
# mc = rfs_table(mbrfs, collect(0:length(xs)), GenRFS.cardinality)
# charges = [1, 1, 3]

# display(ml)
# display(mc)
# p = GenRFS.max_assignment(ml, mc, charges)
# display(p)
# # @btime GenRFS.max_assignment(ml, mc, charges)

# k_ins = GenRFS.ins_kernel(p, ml, mc)
# display(k_ins)

# k_swp = GenRFS.swap_kernel(p, ml)
# display(k_swp)

state = GenRFS.RTWState(mbrfs, xs)
# display(state.partition)
# display(state.k_swp)
# display(state.k_ins)

function random_steps(st, n, t)
    for _ = 1:n
        GenRFS.random_tree_step!(state; t = t)
    end
    return nothing
end
steps = 100
temp = 10.
random_steps(state, steps, temp)
# display(state.partition)
# display(state.k_swp)
# display(state.k_ins)
logscores = collect(values(state.logscores_map))
println("acceptance ratio: $(length(logscores) / (steps+1))")
@show logsumexp(logscores)
wls = exp.(logscores .- logsumexp(logscores))

GenRFS.modify_partition_ctx!(0)
ls, p_tensor = GenRFS.associations(mbrfs, xs)
# top_k = sortperm(ls, rev = true)[1:12]
# for i in top_k
#     pt = p_tensor[:, :, i]
#     hpt = GenRFS.hash_pmat(pt)
#     haskey(state.partition_map, hpt) && continue
#     @show ls[i]
#     display(pt)
# end
@show logsumexp(ls)
# wls = exp.(ls .- logsumexp(ls))
display(histogram(logscores,
                  vertical = true,
                  width = 50,
                  title = "Approximation",
                  xlim = (minimum(ls),
                          maximum(ls))))
display(histogram(ls, vertical = true,
                  bins = 50,
                  width = 50,
                  title = "P(xs | es)"))
display(@benchmark random_steps(state, steps, temp))
display(@benchmark GenRFS.associations(mbrfs, xs))


# state = GenRFS.RTWState(mbrfs, xs)

# function random_steps(st, n)
#     for _ = 1:n
#         GenRFS.random_tree_step!(state)
#     end
#     return nothing
# end
# function prof_function()
#     @profilehtml random_steps(state, 100, 10.)
# end
# Profile.init(delay = 1E-6,
#              n = 10^7)
# Profile.clear()
# prof_function()
# # @btime GenRFS.random_tree_step!(state)
# # display(state.partition)
# # display(state.k_swp)
# # display(state.k_ins)
# state = GenRFS.RTWState(mbrfs, xs)
# random_steps(state, 100)
# logscores = collect(values(state.logscores_map))
# @show length(logscores)
# @show logsumexp(logscores)
# @show logpdf(rfs, xs, mbrfs)
# display(@benchmark logpdf(rfs, xs, mbrfs))
# # @profilehtml logpdf(rfs, xs, mbrfs)
