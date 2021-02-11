export normalize_weights, partition_table

import Base.map # not really sure why we crash without this import
using Base:tail
using LRUCache
using Combinatorics
using IterTools: groupby, firstrest
using Base.Iterators:product, flatten, rest
using DataStructures: DisjointSets, find_root!
using Cassette: Cassette, @context, overdub, recurse

using BenchmarkTools: @btime

const Partition = DisjointSets{Int64}
const PartitionTable = Vector{Partition}

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
    rng = collect(Int64, 1:k)
    @>> pressed begin
        map(x -> partition_push(x, rng))
        x -> vcat(x...)
    end
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

# function filter_bounds(x, u, l)
#     all(((u - x) .>= 0) .& ((x - l) .>= 0))
# end
filter_bounds(x, u, l) = all(((u - x) .>= 0) .& ((x - l) .>= 0))

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

    # obtain the possible cards
    a = filter(x -> length(x) <= nx, integer_partitions(k))
    # 5-element Array{Array{Int64,1},1}:
    #  [1, 1, 1, 1]
    #  [2, 1, 1]
    #  [2, 2]
    #  [3, 1]
    #  [4]
    na = length(a)
    table = fill(0, nx, na)
    # pad cards
    for i = 1:na
        v = a[i]
        table[1:length(v), i] = v
    end

    # remove partitions that have too many or too few assignments for any element
    cfilter = y -> filter(x -> filter_bounds(x, upper, lower), y)
    combs = @>> table begin
        eachcol
        collect(Vector{Int64})
        cfilter
        x -> hcat(x...)
    end

    # getting all permutations of cardinalities
    combs_permuted = @>> combs begin
        eachcol
        map(cfilter ∘ unique ∘ permutations )
        x -> vcat(x...)
    end

    ## The code below is more effecient but not fully tested
    ## for generalization

    # levels = unique(upper)
    # level_idxs = indexin(levels, upper)
    # push!(level_idxs, size(combs, 2))

    # level_perms = Vector{Vector{Int64}}[]
    # beg = 1
    # # for each assignment mapping, construct all permutations
    # for stp in level_idxs
    #     lvl_perm = @>> (combs[:, beg:stp]) begin
    #         eachcol
    #         map(cfilter ∘ unique ∘ permutations)
    #     end
    #     append!(level_perms, lvl_perm)
    #     beg = stp + 1
    # end

    # vcat(level_perms...)
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
function partition_push(cs::Vector{Int64}, xs::Vector{Int64})::PartitionTable
    @assert !isempty(cs)
    c = first(cs)
    rst = collect(rest(cs, 2))
    combs = combinations(xs, c)
    isempty(rst) && return map(assign_to, combs)
    @>> combs begin
        map(y -> push_inner(y, xs, rst))
        (ps -> vcat(ps...))
    end
end

function push_inner(y, xs, rst)
    @>> y begin
        setdiff(xs)
        partition_push(rst)
        map(p -> assign_to!(p, y))
    end
end

"""
Creating a new partition
"""
function assign_to(y::Int64)::Partition
    Partition([y])
end

"""
Creating a new partition
"""
function assign_to(y::Vector{Int64})::Partition
    p = Partition()
    assign_to!(p, y)
end

# """
# Adding a null assignment to an existing partition
# """
# function assign_to!(p::Partition, ::Val{e})::Partition
#     minp = minimum(p) - 1
#     push!(p, minp)
#     return p
# end
"""
Adding to an existing partition
"""
function assign_to!(p::Partition, y::Vector{Int64})::Partition
    foreach(x -> push!(p, x), y)
    # minimum and foldl not defined on empy elements
    minp = isempty(p) ? 0 : minimum(p)
    minp = min(0, minp)
    minp -= 1
    isempty(y) ? push!(p, minp) : foldl((a,b) -> union!(p, a, b), y)
    return p
end

partition_indeces(pt::PartitionTable) = map(partition_indeces, pt)
function partition_indeces(p::Partition)::Vector{Vector{Int64}}
    @>> p begin
        groupby(x -> find_root!(p, x))
        # remove null references
        map(y -> filter(x -> sum(x) > 0, y))
        collect(Vector{Int64})
        reverse
    end
end
