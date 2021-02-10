export normalize_weights, partition_table

import Base.map # not really sure why we crash without this import
using Base:tail
using LRUCache
using Combinatorics
using IterTools: groupby, firstrest
using Base.Iterators:product, flatten, rest
using DataStructures: DisjointSets, find_root!
using Cassette: Cassette, @context, overdub, recurse

function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end


"""Returns the partition table given the morphological bounds of RFEs

Each top level entry is a possible partition. Within each partions,
each entry describes the associations of observation elements to a given element.

The elements are ordered by upper morphological bound in descending order.

> NOTE: `Isomorphic elements are not currently supported`
"""
function partition_table(upper::Vector{Int}, lower::Vector{Int}, k::Int)::PartitionTable
    @assert issorted(upper, rev = true)
    @assert sum(lower) == 0 # no isomorphic elements
    pressed = partition_press(upper, lower, k)
    rng = collect(1:k)
    vcat(map(x -> partition_push(x, rng), pressed)...)
end

# Memoizing the partition table for great fun
@context MemoizeCtx

# TODO: add type sig to LRU
partition_ctx = MemoizeCtx(metadata = LRU(maxsize = 10))

function Cassette.overdub(ctx::MemoizeCtx, ::typeof(partition_table), x, y, z)
    result = get(ctx.metadata, x => y => z, 0)
    if result === 0
        result = recurse(ctx, partition_table, x, y, z)
        ctx.metadata[x => y => z] = result
    end
    return result
end

function mem_partition_table(upper::Vector{Int}, lower::Vector{Int}, k::Int)
    Cassette.overdub(partition_ctx, partition_table, upper, lower, k)
end

"""
Returns the combination table for correspondence cardinality.

For an ordered set of morphological bounds, returns a table
where each row describes a different combination of cardinalities of assignments.

ie

```
julia> GenRFS.partition_press([4,1,1],[0,0,0], 4)
4-element Array{Array{Int64,1},1}:
 [2, 1, 1]
 [3, 1, 0]
 [3, 0, 1]
 [4, 0, 0]
```
"""
function partition_press(upper::Vector{Int}, lower::Vector{Int}, k::Int)
    nx = length(upper)
    # special case with k == 0
    k == 0 && return [collect(Int64, zeros(nx))]

    # remove partitions that require too many elements
    a = filter(x -> length(x) <= nx, integer_partitions(k))
    table = convert.(Int64, zeros(length(a), nx))
    for i = 1:length(a)
        v = a[i]
        table[i, 1:length(v)] = v
    end
    # remove partitions that have too many assignments for any element
    combs = vcat(filter(r -> all(((upper .- r) .>= 0.) .& ((r .- lower) .>= 0.)),
                           collect(eachrow(table)))'...)
    
    # getting all permutations of cardinalities
    combs_permuted = vcat(map(comb -> collect(unique(permutations(comb))), eachrow(combs))...)
    # filtering according to the lower and upper bounds
    combs_filtered = filter(r -> all((upper .- r) .>= 0) && all((r .- lower) .>= 0), combs_permuted)
    return combs_filtered
    
    # old code, I suppose it's more efficient, but need to fix the bug
    # where first component doesn't ever get 0 observations
    # in 2 observations, 3 random finite elements scenario
    
    levels = unique(upper)
    level_idxs = indexin(levels, upper)[2:end]
    push!(level_idxs, nx + 1)

    level_perms = []
    beg = 1
    # for each assignment mapping, construct all permutations
    for l = 1:length(levels)
        stp = level_idxs[l] - 1
        lvl_perm = map(unique ∘ permutations, eachrow(combs[:, beg:stp]))
        push!(level_perms, lvl_perm)
        beg = stp + 1
    end

    # create the full cardinality table across the levels
    pressed = flatten(product.(level_perms...))
    collect(map(x -> vcat(x...), pressed))
end

"""
Returns a table of all associations for a given cardinality combination.

Rows are the complete disjoint partitioning of the elements in `xs` onto
elements described in `cs`

eg.
```
julia> GenRFS.partition_push([2,1,1], [1,2,3,4])
12-element Array{Array{Array{Int64,1},1},1}:
 [[1, 2], [3], [4]]
 [[1, 2], [4], [3]]
 [[1, 3], [2], [4]]
 [[1, 3], [4], [2]]
 [[1, 4], [2], [3]]
 [[1, 4], [3], [2]]
 [[2, 3], [1], [4]]
 [[2, 3], [4], [1]]
 [[2, 4], [1], [3]]
 [[2, 4], [3], [1]]
 [[3, 4], [1], [2]]
 [[3, 4], [2], [1]]
```
"""
function partition_push(cs::Vector{Int64}, xs::Vector{Int64})
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
# function partition_push(cs::Vector{Int64}, xs::Vector{Int64})::PartitionTable
#     result = PartitionTable()
#     isempty(cs) && return result
#     c = first(cs)
#     base = @>> c combinations(xs) collect(Int64)
#     rst = collect(rest(cs, 2))
#     # exit condition
#     for ys in base
#         rems = setdiff(xs, ys)
#         right = partition_push(rst, rems)
#         # TODO look for speed up here?
#         append!(result, [[ys, r...] for r in right])
#     end
#     result
# end

const Partition = DisjointSets{Int64}
const PartitionTable = Vector{Partition}

function foo(
    cs::Vector{Int64},
    xs::Vector{Int64},
)::PartitionTable
    @assert !isempty(cs)
    c = first(cs)
    # @show c
    rst = collect(rest(cs, 2))
    # @show rst
    combs = combinations(xs, c)
    isempty(rst) && return map(assign_to, combs)
    @>> combs begin
        map(y -> foo_inner(y, xs, rst))
        (ps -> vcat(ps...))
    end
end

function foo_inner(y, xs, rst)
    @>> y begin
        setdiff(xs)
        foo(rst)
        map(p -> assign_to!(p, y))
    end
end

function assign_to(y::Int64)::Partition
    Partition([y])
end
function assign_to(y::Vector{Int64})::Partition
    p = Partition(y)
    foldl((a,b) -> union!(p, a, b), y)
    return p
end
function assign_to!(p::Partition, y::Vector{Int64})::Partition
    foreach(x -> push!(p, x), y)
    foldl((a,b) -> union!(p, a, b), y)
    return p
end

partition_indeces(pt::PartitionTable) = map(partition_indeces, pt)
function partition_indeces(p::Partition)::Vector{Vector{Int64}}
    @>> p begin
        groupby(x -> find_root!(p, x))
        collect(Vector{Int64})
        reverse
    end
end

function bar(cs::Vector{Int64}, xs::Vector{Int64})
    c = first(cs)
    ys = combinations(xs, c)
    rst = collect(rest(cs, 2))
    @>> ys begin
        map(y -> foo(xs, y, rst))
    end
end
