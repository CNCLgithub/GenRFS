using SHA
using LinearAlgebra
using DataStructures

function sample_partition_cube(l_table::Matrix{Float64},
                               c_table::Matrix{Float64},
                               max_charges::Vector{Int64})

end

function max_assignment(l_table::Matrix{Float64},
                        c_table::Matrix{Float64},
                        max_charges::Vector{Int64})::BitMatrix
    partition = zeros(Bool, size(l_table'))
    max_ls = vec(maximum(l_table, dims = 1))
    # start with the "closest" assignment
    @inbounds for xi = sortperm(max_ls, rev = true)
        # filter out elements that have an upper bound
        free_elements = findall(vec(sum(partition, dims = 1)) .< max_charges)
        # index magic
        ei = free_elements[argmax(l_table[free_elements, xi])]
        partition[xi, ei] = true
    end
    BitMatrix(partition)
end

function ins_kernel(partition::BitMatrix, l_table::Matrix{Float64}, c_table::Matrix{Float64})
    (ne, nx) = size(l_table)
    k_ins = fill(-Inf, (nx, ne))
    ins_kernel!(k_ins, partition, l_table, c_table)
    return k_ins
end
function ins_kernel!(k_ins::Matrix{Float64},
                    partition::BitMatrix,
                    l_table::Matrix{Float64},
                    c_table::Matrix{Float64})::Nothing
    (ne, nx) = size(l_table)
    ecs = count.(eachcol(partition)) .+ 1
    partition = Matrix{Bool}(partition')
    # total number of assignments per element
    # inc by one for indeces in c_table
    @inbounds for x = 1:nx
        # currently assigned element
        ei = findfirst(view(partition, :, x))
        eil = l_table[ei, x]
        eic = c_table[ei, ecs[ei] - 1]
        log_denom = (eil + eic)
        for ej = 1:ne
            ej == ei && continue
            ejc = ecs[ej]
            k_ins[x, ej] = l_table[ej, x] + c_table[ej, ejc+1]  - log_denom
        end
    end
    return nothing
end

function swap_kernel(partition::BitMatrix, l_table::Matrix{Float64})
    (ne, nx) = size(l_table)
    k_swap = fill(-Inf, (nx, nx))
    swap_kernel!(k_swap, partition, l_table)
    return k_swap
end
function swap_kernel!(k_swap::Matrix{Float64},
                      partition::BitMatrix,
                      l_table::Matrix{Float64})::Nothing
    partition = Matrix{Bool}(partition')
    (ne, nx) = size(l_table)
    @inbounds for a = 1:nx
        # currently assigned element
        ei = findfirst(view(partition, :, a))
        laei = l_table[ei, a]
        for b = 1:nx
            a <= b && continue
            ej = findfirst(view(partition, :, b))
            if ei == ej
                # can't swap when assigned to same element
                k_swap[b, a] = -Inf
                continue
            end
            lbej = l_table[ej, b]
            lbei = l_table[ei, b]
            laej = l_table[ej, a]
            # upper triangle
            k_swap[b, a] = (laej - laei) + (lbei - lbej)
            # if iszero(k_swap[b, a])
            #     @show a => b
            #     @show lbej
            #     @show lbei
            #     @show laej
            #     @show laei
            # end
        end
    end
    return nothing
end

function partition_score(partition::BitMatrix, ml::Matrix{Float64}, mc::Matrix{Float64})::Float64
    part_ls = 0.0
    partition = Matrix{Bool}(partition)
    nx,ne = size(partition)
    @inbounds for e = 1:ne
        part_ls === -Inf && break # no need to continue if -Inf
        nassoc = 1
        assoc_ls = 0.0
        for x = 1:nx
            partition[x, e] || continue
            nassoc += 1
            part_ls += ml[e, x]
        end
        part_ls += mc[e, nassoc]
    end
    return part_ls
end

struct RandomTreeWalk end

mutable struct RTWState
    ml::Matrix{Float64}
    mc::Matrix{Float64}
    partition::BitMatrix
    k_swp::Matrix{Float64}
    k_ins::Matrix{Float64}
    partition_map::Dict{String, BitMatrix}
    logscores_map::Dict{String, Float64}
end

function RTWState(es::RFSElements{T}, xs::Vector{T}) where {T}
    ml = rfs_table(es, xs, support)
    mc = rfs_table(es, collect(0:length(xs)), cardinality)
    us = Int64.(clamp.(upper.(es), 0, length(xs)))
    #start off with arbitrary partition
    pstart = max_assignment(ml, mc, us)
    ls = partition_score(pstart, ml, mc)
    # initialize kernels
    k_swp = swap_kernel(pstart, ml)
    k_ins = ins_kernel(pstart, ml, mc)
    # add entries to queues
    hs = hash_pmat(pstart)
    pm = Dict{String, BitMatrix}(hs => pstart)
    lm = Dict{String, Float64}(hs => ls)
    RTWState(ml, mc, pstart, k_swp, k_ins, pm, lm)
end

function hash_pmat(pmat::BitArray)
    bytes2hex(sha256(reinterpret(UInt8, pmat.chunks)))
end


function swap_move!(st::RTWState, a::Int, b::Int)::Nothing
    # p = Matrix{Bool}(st.partition)
    p = st.partition
    be = findfirst(view(p, b, :))
    ae = findfirst(view(p, a, :))
    st.partition[b, be] = false
    st.partition[b, ae] = true
    st.partition[a, ae] = false
    st.partition[a, be] = true
    return nothing
end

function insert_move!(st::RTWState, x::Int, e::Int)::Nothing
    xe = findfirst(view(st.partition, x, :))
    st.partition[x, xe] = false
    st.partition[x, e] = true
    return nothing
end

function update_from_move!(st::RTWState)
    #TODO: is it faster to shift values around
    # for swap and insert moves independently
    swap_kernel!(st.k_swp, st.partition, st.ml)
    ins_kernel!(st.k_ins, st.partition, st.ml, st.mc)

    hs = hash_pmat(st.partition)
    pt = BitMatrix(st.partition)
    st.partition_map[hs] = pt
    st.logscores_map[hs] = partition_score(pt, st.ml, st.mc)
    return nothing
end

function greedy_tree_step!(st::RTWState)::Nothing
    max_swpi = argmax(st.k_swp)
    max_insi = argmax(st.k_ins)
    if st.k_swp[max_swpi] > st.k_ins[max_insi]
        # swap move
        (b, a) = Tuple(max_swpi)
        swap_move!(st, a, b)
    else
        # insertion
        (x, e) = Tuple(max_swpi)
        insert_move!(st, x, e)
    end
    update_from_move!(st)
    return nothing
end


function softmax(x::Array{Float64}; t::Float64 = 1.0)
    out = similar(x)
    softmax!(out, x; t = t)
    return out
end

function softmax!(out::Array{Float64}, x::Array{Float64}; t::Float64 = 1.0)
    nx = length(x)
    maxx = maximum(x)
    sxs = 0.0

    if maxx == -Inf
        out .= 1.0 / nx
        return nothing
    end

    @inbounds for i = 1:nx
        out[i] = @fastmath exp((x[i] - maxx) / t)
        sxs += out[i]
    end
    rmul!(out, 1.0 / sxs)
    return nothing
end

function random_tree_step!(st::RTWState;
                           t::Float64 = 1.0)::Nothing
    pswp = softmax(st.k_swp, t = t)
    swpi = categorical(vec(pswp))
    pins = softmax(st.k_ins, t = t)
    insi = categorical(vec(pins))
    if st.k_swp[swpi] > st.k_ins[insi]
        # swap move
        b = Int(((swpi-1) % size(st.k_swp, 1)) + 1)
        a = Int(ceil(swpi / size(st.k_swp, 2)))
        swap_move!(st, a, b)
    else
        # insertion
        nx = size(st.k_ins, 1)
        (x, e) = Int(((insi-1) % nx) + 1), Int(ceil(insi / nx))
        insert_move!(st, x, e)
    end
    update_from_move!(st)
    return nothing
end
