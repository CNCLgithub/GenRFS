export normalize_weights, partition_press

import Base.map # not really sure why we crash without this import
using Base:tail
using Base.Iterators:product, flatten, rest
using Memoize

function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

function partition_table(upper::Vector{Int}, lower::Vector{Int}, k::Int)
    @assert issorted(upper, rev = true)
    pressed = partition_press(upper, lower, k)
    rng = collect(1:k)
    vcat(map(x -> partition_push(x, rng), pressed)...)
end

using Cassette: Cassette, @context, overdub, recurse

@context MemoizeCtx

function Cassette.overdub(ctx::MemoizeCtx, ::typeof(partition_table), x, y, z)
    result = get(ctx.metadata, x => y => z, 0)
    if result === 0
        result = recurse(ctx, partition_table, x, y, z)
        ctx.metadata[x => y => z] = result
    end
    return result
end

function partition_press(upper::Vector{Int}, lower::Vector{Int}, k::Int)
    nx = length(upper)
    a = filter(x -> length(x) <= nx,
               integer_partitions(k))
    table = convert.(Int64, zeros(length(a), nx))
    for i = 1:length(a)
        v = a[i]
        table[i, 1:length(v)] = v
    end
    combs = vcat(filter(r -> all(((upper .- r) .>= 0.) .& ((r .- lower) .>= 0.)),
                           collect(eachrow(table)))'...)
    levels = unique(upper)
    level_idxs = indexin(levels, upper)[2:end]
    push!(level_idxs, nx + 1)
    level_perms = []
    beg = 1
    for l = 1:length(levels)
        stp = level_idxs[l] - 1
        lvl_perm = map(unique ∘ permutations,
                            eachrow(combs[:, beg:stp]))
        push!(level_perms, lvl_perm)
        beg = stp + 1
    end
    pressed = flatten(product.(level_perms...))
    collect(map(x -> vcat(x...), pressed))
end

function partition_push(cs::Vector{Int}, xs::Vector{Int})
    @assert !isempty(cs)
    c = first(cs)
    n = length(cs)
    base = collect(combinations(xs, c))
    rst = collect(rest(cs, 2))
    # exit condition
    isempty(rst) && return [base]
    result = collect(Vector{Vector{Int64}}, [])
    for ys in base
        rems = setdiff(xs, ys)
        right = partition_push(rst, rems)
        # TODO look for speed up here?
        append!(result, [[ys, r...] for r in right])
    end
    result
end
