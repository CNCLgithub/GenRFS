export normalize_weights, partition_table, modify_partition_ctx!

import Base.map # not really sure why we crash without this import
using Base:tail
using LRUCache
using Combinatorics
using IterTools: groupby, firstrest
using Base.Iterators:product, flatten, rest
using DataStructures: DisjointSets, find_root!
using Cassette: Cassette, @context, overdub, recurse

using LightGraphs
using FunctionalCollections: pvec, assoc, PersistentVector

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
function partition_table(upper::Vector{Int}, lower::Vector{Int}, k::Int)
    @assert issorted(upper, rev = true)
    @assert sum(lower) == 0 # no isomorphic elements
    pressed = partition_press(upper, lower, k)
    rng = collect(Int64, 1:k)
    @>> pressed begin
        map(x -> partition_push(x, rng))
        x -> vcat(x...)
        partition_indeces
    end
end

# Memoizing the partition table for great fun
# function filter_bounds(x, u, l)
#     all(((u - x) .>= 0) .& ((x - l) .>= 0))
# end
filter_bounds(x, u, l) = all(((u - x) .>= 0) .& ((x - l) .>= 0))


mutable struct RFTree
    g::SimpleDiGraph
    max_depth::Int64
    es::Dict{Int64, Int64}
    leaves::Vector{Int64}
    RFTree(nx) = new(SimpleDiGraph(), nx,
                     Dict{Int64, Int64}(),
                     Int64[])
end

function partition_cube(a_table::BitMatrix, max_charges::Vector{Int64})
    ne, nx = size(a_table)
    tree = init_tree(nx)
    # BFS
    walk_tree!(tree, pvec(max_charges), a_table)
    # convert to bit cube
    cube_from_tree(tree, ne, nx)
end

function init_tree(nx::Int64)
    tree = RFTree(nx)
    g = tree.g
    # create root node @ 1
    add_vertex!(g)
    return tree
end

function walk_tree!(tree::RFTree, charges::PersistentVector, a_table::BitMatrix)
    moves = findall(a_table[:, 1] .& (charges .> 0))
    for m in moves
        walk_tree!(tree, 1, 1, charges, a_table, m)
    end
    return nothing
end
function walk_tree!(tree::RFTree, loc::Int64, depth::Int64,
                    charges::PersistentVector, a_table::BitMatrix,
                    m::Int64)
    g = tree.g
    # add new node
    add_vertex!(g)
    v = nv(g)
    tree.es[v] = m
    add_edge!(g, loc, v)

    # check to see if done
    if depth == tree.max_depth
        push!(tree.leaves, v)
        return nothing
    end

    # decrement from charges
    remaining = assoc(charges, m, charges[m] - 1)

    # figure out next moves
    moves = findall(a_table[:, depth] .& (remaining .> 0))
    for m in moves
        walk_tree!(tree, v, depth + 1, remaining, a_table, m)
    end
    return nothing
end

function cube_from_tree(tree, ne, nx)
    g = tree.g
    es = tree.es
    leaves = tree.leaves
    nl = length(leaves)
    bc = falses(nx, ne, nl)
    @inbounds for i = 1:nl
        v = leaves[i]
        x = nx
        while v != 1
            bc[x, es[v], i] = true
            # inc
            v = first(inneighbors(g, v))
            x -= 1
        end
    end
    return bc
end

@context MemoizeCtx

# TODO: add type sig to LRU
partition_ctx = MemoizeCtx(metadata = LRU(maxsize = 10))

function modify_partition_ctx!(maxsize::Int64)
    global partition_ctx = MemoizeCtx(metadata = LRU(maxsize = maxsize))
end

function Cassette.overdub(ctx::MemoizeCtx, ::typeof(partition_cube), x, y)
    # add ability to ignore cache
    typeof(ctx.metadata) == LRU{Any, Any} &&
        ctx.metadata.maxsize == 0 &&
        return partition_cube(x, y)
    result = get(ctx.metadata, x => y, 0)
    if result === 0
        result = recurse(ctx, partition_cube, x, y)
        ctx.metadata[x => y] = result
    end
    return result
end

function mem_partition_cube(a_table::BitMatrix, max_charges::Vector{Int64})
    Cassette.overdub(partition_ctx, partition_cube, a_table, max_charges)
end
