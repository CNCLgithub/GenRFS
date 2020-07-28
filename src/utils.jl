export normalize_weights, partition_press

using Base:tail
using Base.Iterators:product

function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

function partition_press(m::Vector{Int}, k::Int)
    m = sort(m, rev = true)
    nx = length(m)
    a = filter(x -> length(x) <= nx,
               integer_partitions(k))
    table = convert.(Int64, zeros(length(a), nx))
    for i = 1:length(a)
        v = a[i]
        table[i, 1:length(v)] = v
    end
    combs = vcat(filter(r -> all((m .- r) .>= 0.),
                           collect(eachrow(table)))'...)
    display(combs)
    levels = unique(m)
    level_idxs = indexin(levels, m)[2:end]
    push!(level_idxs, nx + 1)
    level_perms = []
    beg = 1
    for l = 1:length(levels)
        stp = level_idxs[l] - 1
        println("$beg -> $stp")
        display(combs[:, beg:stp])
        lvl_perm = map(unique ∘ permutations , eachrow(combs[:, beg:stp]))
        display(lvl_perm)
        push!(level_perms, lvl_perm)
        beg = stp + 1
    end
    collect(product.(zip(level_perms...)))
end
