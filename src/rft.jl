export mem_partition_cube

using LRUCache
using FunctionalCollections: pvec, assoc, PersistentVector

function partition_cube(a_table::BitMatrix, max_charges::Vector{Int})
    ne, nx = size(a_table)
    nx === 0 && return falses(0, ne, 1)
    
    # Pre-allocate output buffer
    partitions = Vector{NTuple{nx, UInt16}}()
    assignment = zeros(UInt16, nx)
    charges = copy(max_charges)
    
    # DFS Recursive Backtracking
    function dfs(depth::Int)
        if depth > nx
            push!(partitions, Tuple(assignment))
            return
        end
        
        for e in 1:ne
            if a_table[e, depth] && charges[e] > 0
                assignment[depth] = UInt16(e)
                charges[e] -= 1
                dfs(depth + 1)
                charges[e] += 1 # Backtrack
            end
        end
    end
    
    dfs(1)
    
    # Convert partitions vector to 3D BitArray
    np = length(partitions)
    np == 0 && return falses(nx, ne, 0)
    
    p_cube = falses(nx, ne, np)
    @inbounds for (p, tup) in enumerate(partitions)
        for x in 1:nx
            p_cube[x, Int(tup[x]), p] = true
        end
    end
    
    return p_cube
end

# mutable struct RFTree
#     g::SimpleDiGraph
#     max_depth::Int64
#     es::Dict{Int64, Int64}
#     leaves::Vector{Int64}
#     RFTree(nx) = new(SimpleDiGraph(),
#                      nx,
#                      Dict{Int64, Int64}(),
#                      Int64[])
# end

# function partition_cube(a_table::BitMatrix, max_charges::Vector{Int64})
#     ne, nx = size(a_table)
#     tree = init_tree(nx)
#     # BFS
#     walk_tree!(tree, pvec(max_charges), a_table)
#     # convert to bit cube
#     cube_from_tree(tree, ne, nx)
# end

# function init_tree(nx::Int64)
#     tree = RFTree(nx)
#     g = tree.g
#     # create root node @ 1
#     add_vertex!(g)
#     return tree
# end

# function walk_tree!(tree::RFTree, charges::PersistentVector, a_table::BitMatrix)
#     moves = findall(a_table[:, 1] .& (charges .> 0))
#     for m in moves
#         walk_tree!(tree, 1, 1, charges, a_table, m)
#     end
#     nothing
# end
# function walk_tree!(tree::RFTree, loc::Int64, depth::Int64,
#                     charges::PersistentVector, a_table::BitMatrix,
#                     m::Int64)
#     g = tree.g
#     # add new node
#     add_vertex!(g)
#     v = nv(g)
#     tree.es[v] = m
#     add_edge!(g, loc, v)
#     # check to see if done
#     if depth == tree.max_depth
#         push!(tree.leaves, v)
#         return nothing
#     end
#     # decrement from charges
#     remaining = assoc(charges, m, charges[m] - 1)
#     # figure out next moves
#     moves = findall(a_table[:, depth + 1] .& (remaining .> 0))
#     for m in moves
#         walk_tree!(tree, v, depth + 1, remaining, a_table, m)
#     end
#     nothing
# end

# function cube_from_tree(tree::RFTree, ne::Int64, nx::Int64)
#     g = tree.g
#     es = tree.es
#     leaves = tree.leaves
#     nl = length(leaves)
#     bc = falses(nx, ne, nl)
#     @inbounds for i = 1:nl
#         v = leaves[i]
#         x = nx
#         while v != 1
#             bc[x, es[v], i] = true
#             # inc
#             v = first(inneighbors(g, v))
#             x -= 1
#         end
#     end
#     return bc
# end

# const CTX_Key = Pair{BitMatrix, Vector{Int64}}
# const CTX_Val = BitArray{3}
# const CTX_CACHE = LRU{CTX_Key, CTX_Val}

const PARTITION_CACHE = LRU{Pair{BitMatrix, Vector{Int}}, BitArray{3}}(maxsize=100)

function mem_partition_cube(a_table::BitMatrix, max_charges::Vector{Int})
    key = a_table => max_charges
    get!(PARTITION_CACHE, key) do
        partition_cube(a_table, max_charges)
    end
end

# @context MemoizeCtx
# partition_ctx = MemoizeCtx(metadata = CTX_CACHE(maxsize=100))

# function modify_partition_ctx!(maxsize::Int64)
#     global partition_ctx = MemoizeCtx(metadata = CTX_CACHE(maxsize=maxsize))
#     nothing
# end


# function Cassette.overdub(ctx::MemoizeCtx, ::typeof(partition_cube),
#                           x::BitMatrix, y::Vector{Int64})::BitArray{3}
#     result = get(ctx.metadata, x => y, 0)
#     if result === 0
#         result = partition_cube(x, y)
#         ctx.metadata[x => y] = result
#     end
#     return result
# end

# function mem_partition_cube(a_table::BitMatrix, max_charges::Vector{Int64})
#     Cassette.overdub(partition_ctx, partition_cube,
#                      a_table, max_charges)
# end

