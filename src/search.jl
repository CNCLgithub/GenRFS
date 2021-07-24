using LightGraphs
using SparseArrays
using FunctionalCollections: pvec, assoc, PersistentVector

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
