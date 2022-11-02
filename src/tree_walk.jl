using SHA
using LinearAlgebra
using DataStructures


function support_table(es::RFSElements{T},
                       xs::Vector{T})::Matrix{Float64} where {T}
    nx = length(xs)
    ne = length(es)
    table = Matrix{Float64}(undef, ne, nx)
    @inbounds for ei = 1:ne, xi = 1:nx
        table[ei, xi] = support(es[ei],xs[xi])
    end
    table
end
function cardinality_table(es::RFSElements{T},
                       xs::Vector{T})::Matrix{Float64} where {T}
    nx = length(xs)
    ne = length(es)
    table = Matrix{Float64}(undef, ne, nx + 1)
    @inbounds for ei = 1:ne, xi = 0:nx
        table[ei, xi+1] = cardinality(es[ei], xi)
    end
    table
end

function max_assignment(l_table::Matrix{Float64},
                        c_table::Matrix{Float64},
                        max_charges::Vector{Int64})::BitMatrix
    partition = zeros(Bool, size(l_table'))
    max_ls = vec(maximum(l_table, dims = 1))
    # start with the "closest" assignment
    @inbounds @views for xi = sortperm(max_ls, rev = true)
        # prefer most restricted elements in terms of constraints
        for ei = sortperm(max_charges)
            count(partition[:, ei]) >= max_charges[ei] && continue
            partition[xi, ei] = true
            break
        end
    end
    BitMatrix(partition)
end

function upper_t_size(n::Int64)
    Int64(n * (n-1) / 2)
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
    # number of assignments per element
    ecs = count.(eachcol(partition)) .+ 1
    # partition = Matrix{Bool}(partition')
    @inbounds @views for x = 1:nx
        # currently assigned element
        ei = findfirst(partition[x, :])
        pxei = l_table[ei, x]
        pci = c_table[ei, ecs[ei]]
        for ej = 1:ne
            k_ins[x, ej] = if ej == ei
                # don't reassign
                -Inf
            else
                # k_ins = prob new assignment / prob current assign
                # P(x | ej) * P(c_j + 1) * P(c_i - 1) /
                # P(x | ei) * P(c_j)     * P(ci)
                pcj     = c_table[ej, ecs[ej]]
                pci_dec = c_table[ei, ecs[ei] - 1]
                pcj_inc = c_table[ej, ecs[ej] + 1]
                pxej = l_table[ej, x]
                k_ins[x, ej] = (pxej + pcj_inc + pci_dec) -
                    (pxei + pcj + pci)
            end
        end
    end
    return nothing
end

function swap_kernel(partition::BitMatrix, l_table::Matrix{Float64})
    (ne, nx) = size(l_table)
    k_swap = Vector{Float64}(undef, upper_t_size(nx))
    swap_kernel!(k_swap, partition, l_table)
    return k_swap
end
function swap_kernel!(k_swap::Vector{Float64},
                      partition::BitMatrix,
                      l_table::Matrix{Float64})::Nothing
    # partition = Matrix{Bool}(partition')
    (ne, nx) = size(l_table)
    @assert upper_t_size(nx) == length(k_swap) "swap kernel size missmatch"
    nx == 0 && return nothing
    i = 0
    @inbounds @views for a = 1:(nx - 1)
        # currently assigned element
        # ei = findfirst(partition[:, a])
        ei = findfirst(partition[a, :])
        laei = l_table[ei, a]
        for b = (a+1):nx
            i += 1
            # ej = findfirst(view(partition, :, b))
            ej = findfirst(partition[b, :])
            if ei == ej
                # can't swap when assigned to same element
                k_swap[i] = -Inf
                continue
            end
            lbej = l_table[ej, b]
            lbei = l_table[ei, b]
            laej = l_table[ej, a]
            # upper triangle
            # P(a|ej) * P(b|ei) / P(a|ei) * P(b|ej)
            k_swap[i] = (laej + lbei) - (laei + lbej)
        end
    end
    return nothing
end

function partition_score(partition::BitMatrix, ml::Matrix{Float64}, mc::Matrix{Float64})::Float64
    part_ls = 0.0
    # partition = Matrix{Bool}(partition)
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
    pscore::Float64
    k_swp::Vector{Float64}
    k_ins::Matrix{Float64}
    nk_swp::Vector{Float64}
    nk_ins::Matrix{Float64}
    partition_map::Dict{BitMatrix, Float64}
end

function RTWState(es::RFSElements{T}, xs::Vector{T}) where {T}
    ml = support_table(es, xs)
    mc = cardinality_table(es, xs)
    us = Int64.(clamp.(upper.(es), 0, length(xs)))
    #start off with arbitrary partition
    pstart = max_assignment(ml, mc, us)
    ls = partition_score(pstart, ml, mc)
    # initialize kernels
    k_swp = swap_kernel(pstart, ml)
    k_ins = ins_kernel(pstart, ml, mc)
    # normalized kernels
    nk_swp = Vector{Float64}(undef, length(k_swp))
    nk_ins = Matrix{Float64}(undef, size(k_ins))
    # add entries to queues
    # dereference initial partition
    pm = Dict{BitMatrix, Float64}(BitMatrix(pstart) => ls)
    RTWState(ml, mc, pstart, ls, k_swp, k_ins, nk_swp, nk_ins, pm)
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

function update_from_move!(st::RTWState, w::Float64)
    # update kernels
    swap_kernel!(st.k_swp, st.partition, st.ml)
    ins_kernel!(st.k_ins, st.partition, st.ml, st.mc)
    # current partition already visited
    haskey(st.partition_map, st.partition) && return nothing
    # dereference new key
    pt = BitMatrix(st.partition)
    # increment score
    pscore = st.pscore + w
    st.partition_map[pt] = pscore
    st.pscore = pscore
    # st.partition_map[pt] = partition_score(pt, st.ml, st.mc)
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
    isempty(x) && return x
    nx = length(x)
    maxx = maximum(x)
    sxs = 0.0

    if maxx == -Inf
        out[:] .= 1.0 / nx
        return nothing
    end

    @inbounds for i = 1:nx
        v = @fastmath exp((x[i] - maxx) / t)
        sxs += v
        out[i] = v
    end
    rmul!(out, 1.0 / sxs)
    return nothing
end

# adapted from
# https://stackoverflow.com/a/68581180
function upper_t_to_matrix(k::Int64, n::Int64)
    i = n - 1 - floor(Int,sqrt(-8*k + 4*n*(n-1) + 1)/2 - 0.5)
    j = k + i + ( (n-i+1)*(n-i) - n*(n-1) )÷2
    return i, j
end



function random_tree_step!(st::RTWState;
                           t::Float64 = 1.0)::Nothing

    mx_kins = maximum(st.k_ins)
    if isinf(mx_kins) || isnan(mx_kins)
        insi = 0
        pins = -Inf
    else
        softmax!(st.nk_ins, st.k_ins, t = t)
        insi = categorical(vec(st.nk_ins))
        pins = st.k_ins[insi]
    end

    # swap kernel could be empty if 1 obs
    if isempty(st.k_swp)
        swpi = 0
        pswap = -Inf
    else
        softmax!(st.nk_swp, st.k_swp, t = t)
        swpi = categorical(st.nk_swp)
        pswap = st.k_swp[swpi]
    end

    # case where no valid moves left
    if isinf(pswap) && isinf(pins)
        update_from_move!(st, 0.)
        return nothing
    end

    nx = size(st.k_ins, 1)
    w = if pswap >= pins
        # swap move
        a, b = upper_t_to_matrix(swpi, nx)
        swap_move!(st, a, b)
        pswap
    else
        # insertion
        (x, e) = Int(((insi-1) % nx) + 1), Int(ceil(insi / nx))
        insert_move!(st, x, e)
        pins
    end
    update_from_move!(st, w)
    return nothing
end
