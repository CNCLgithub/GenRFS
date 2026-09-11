using LinearAlgebra
# using DataStructures # TODO: Remove

"""
    RTWState{K}(...)

The random walk state over partition space
"""
mutable struct RTWState{K}
    ml::Matrix{Float64}
    mc::Matrix{Float64}
    partition::BitMatrix
    pscore::Float64
    k_swp::Vector{Float64}
    k_ins::Matrix{Float64}
    nk_swp::Vector{Float64}
    nk_ins::Matrix{Float64}
    assigned::Vector{Int}
    rowbuf::Vector{Float64}  
    visited::Dict{K, Float64}
end

#--------------------------------------------------------------------------------
# TREE WALK
#--------------------------------------------------------------------------------

"Composed MCMC step: swaps w.p. p_swap (ergodicity under finite-support cardinality), else Gibbs insert."
function mcmc_tree_step!(st::RTWState, t::Float64=1.0, p_swap::Float64=0.5)::Nothing
    if rand() < p_swap
        # MH move swapping ej -> ei
        metro_swap!(st, t)
    else
        # Gibbs move, reassigning x_i from e_j to e_k
        gibbs_insert!(st, t)
    end
    return nothing
end

"One Gibbs insert step: exact conditional over e_x given the rest."
function gibbs_insert!(st::RTWState, t::Float64)::Nothing
    nx = size(st.k_ins, 1)
    # Sample x uniformly
    x  = rand(1:nx)
    ei = st.assigned[x]
    # Sample new ej according to Gibbs
    @inbounds for j = 1:size(st.rowbuf, 1)
        st.rowbuf[j] = (j == ei) ? 0.0 : exp(st.k_ins[x, j] / t)
    end
    ej = unsafe_categorical!(st.rowbuf)
    if ej != ei
        partition_insert_move!(st, x, ei, ej)
        update_after_insert!(st, x, ei, ej)
    end
    return nothing
end

"One Metropolis swap step: uniform proposal over all pairs (state-independent q)."
function metro_swap!(st::RTWState, t::Float64)::Nothing
    nx = size(st.partition, 1)
    N  = nx * (nx - 1) ÷ 2
    i  = rand(1:N)
    a, b = upper_to_pair(i, nx)
    ea, eb = st.assigned[a], st.assigned[b]
    eb == ea && return nothing           # no-op, accepted trivially; q is state-independent
    w = st.k_swp[upper_index(a, b, nx)]  # exact Δscore; move is self-inverse
    if log(rand()) < w / t               # symmetric proposal ⇒ plain Metropolis
        partition_swap_move!(st, a, b)
        update_after_swap!(st, a, b)
    end
    return nothing
end


#--------------------------------------------------------------------------------
# Initialization
#--------------------------------------------------------------------------------

function RTWState(es::RFSElements{T}, xs::AbstractVector{T}) where {T}
    ml = support_table(es, xs)
    mc = cardinality_table(es, xs)
    us = Int64.(clamp.(upper.(es), 0, length(xs)))
    # start off with arbitrary partition
    # - pstart: BitMatrix
    # - assigned: Vector{Int} used for fast inserts
    pstart, assigned = max_assignment(ml, mc, us)
    ls = partition_score(pstart, ml, mc)
    # initialize kernels given initial partition
    k_swp = swap_kernel(pstart, ml)
    k_ins = ins_kernel(pstart, ml, mc)
    # normalized kernels
    # REVIEW: Not used; application for Roa-blackwellized
    # estimation
    nk_swp = Vector{Float64}(undef, length(k_swp))
    nk_ins = Matrix{Float64}(undef, size(k_ins))

    # used in softmaxs; instantiated later;
    ne = length(es)
    rowbuf = Vector{Float64}(undef, ne)
    
    # add entries to queues
    # dereference initial partition
    K = NTuple{length(xs), UInt16}
    pm = Dict{K, Float64}(partition_to_tuple(pstart) => ls)
    state = RTWState{K}(ml, mc, pstart, ls, k_swp, k_ins,
                        nk_swp, nk_ins, assigned, rowbuf, pm)

end

"""
    max_assigment(l, c, m)

Generates a Hungarian-like partition, returning both matrix and vector forms.

Arguments:

- `l_table`: Support table (nx by ne) denoting Pr(x_i | e_j)
- `c_table`: Cardinality table (nx+1 by ne) denoting Pr(c=i | e_j), i=0:nx
- `max_charges`: Maximum non-zero cardinality
"""
function max_assignment(l_table::Matrix{Float64},
                        c_table::Matrix{Float64},
                        max_charges::Vector{Int64}
                        )::Tuple{BitMatrix, Vector{Int64}}
    nx, ne = size(l_table)
    max_ls = vec(maximum(l_table, dims = 1))
    partition = zeros(Bool, size(l_table'))
    assigned = Vector{Int64}(undef, nx)
    # start with the "closest" assignment
    @inbounds @views for xi = sortperm(max_ls, rev = true)
        # prefer most restricted elements in terms of constraints
        for ei = sortperm(max_charges)
            count(partition[:, ei]) >= max_charges[ei] && continue
            partition[xi, ei] = true
            assigned[x] = ei
            break
        end
    end
    (BitMatrix(partition), assigned)
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
    # ecs = count.(eachcol(partition)) .+ 1
    ecs = count_assocs(partition)
    # partition = Matrix{Bool}(partition')
    @inbounds @views for x = 1:nx
        # currently assigned element
        ei = unsafe_find_true(partition[x, :])
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

    # 0 log weight transition to self
    @inbounds for x = 1:nx
        st.k_ins[x, st.assigned[x]] = 0.0
        # REVIEW: could add for RB
        # st.nk_ins[x] = logsumexp(view(st.k_ins, x, :)) 
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
        ei = unsafe_find_true(partition[a, :])
        laei = l_table[ei, a]
        for b = (a+1):nx
            i += 1
            # ej = findfirst(view(partition, :, b))
            ej = unsafe_find_true(partition[b, :])
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


#--------------------------------------------------------------------------------
# PARTITION OPERATIONS
#--------------------------------------------------------------------------------

function partition_insert_move!(st::RTWState, x::Int, ei::Int, ej::Int)::Nothing
    st.partition[x, ei] = false
    st.partition[x, ej] = true
    st.assigned[x] = ej
    st.pscore += st.k_ins[x, ej]
    return nothing
end

function partition_swap_move!(st::RTWState, a::Int, b::Int)::Nothing
    ea, eb = st.assigned[a], st.assigned[b]
    st.partition[a, ea] = false; st.partition[a, eb] = true
    st.partition[b, eb] = false; st.partition[b, ea] = true
    st.assigned[a], st.assigned[b] = eb, ea
    st.pscore += st.k_swp[upper_index(a, b, size(st.partition, 1))]
    return nothing
end

#--------------------------------------------------------------------------------
# MOVE BOOKKEEPING
#--------------------------------------------------------------------------------

"Restore kernel consistency after insert move x: ei -> ej."
function update_after_insert!(st::RTWState, x::Int, ei::Int, ej::Int)::Nothing
    refresh_k_ins_cols!(st, x, ei, ej)   # O(2*nx + ne)
    refresh_k_swp_pairs!(st, x)          # O(nx): pairs involving x
    idx = partition_to_tuple(st.partition)
    haskey(st.visited, idx) || (st.visited[idx] = st.pscore)
    return nothing
end

"Restore kernel consistency after swap (a, b). Counts unchanged; identities changed."
function update_after_swap!(st::RTWState, a::Int, b::Int)::Nothing
    refresh_k_ins_row!(st, a)            # O(ne) each
    refresh_k_ins_row!(st, b)
    refresh_k_swp_pairs!(st, a)          # O(nx) each
    refresh_k_swp_pairs!(st, b)
    idx = partition_to_tuple(st.partition)
    haskey(st.visited, idx) || (st.visited[idx] = st.pscore)
    return nothing
end

#--------------------------------------------------------------------------------
# KERNEL REFRESH PROCEDURES
#--------------------------------------------------------------------------------

"Recompute row x of k_ins. Called when x's element identity changes."
function refresh_k_ins_row!(st::RTWState, x::Int)::Nothing
    ne = size(st.k_ins, 2)
    ej = st.assigned[x]                  # x's current (post-move) element
    c_ej = count_col(st.partition, ej)
    @inbounds for e = 1:ne
        if e == ej
            st.k_ins[x, e] = 0.0
        else
            c_e = count_col(st.partition, e)
            st.k_ins[x, e] = (st.ml[e, x] + st.mc[e, c_e + 1] + st.mc[ej, c_ej - 1])
                           - (st.ml[ej, x] + st.mc[e, c_e] + st.mc[ej, c_ej])
        end
    end
    # nk_ins[x]: row normalizer — byproduct, free ingredient for the RB estimator
    # st.nk_ins[x] = logsumexp(view(st.k_ins, x, :))
    return nothing
end

"Recompute columns ei and ej of k_ins, plus row x (x just moved ei -> ej)."
function refresh_k_ins_cols!(st::RTWState, x::Int, ei::Int, ej::Int)::Nothing
    nx = size(st.k_ins, 1)
    ci = count_col(st.partition, ei)     # post-move counts
    cj = count_col(st.partition, ej)
    @inbounds for x2 = 1:nx
        e_x = st.assigned[x2]
        for e in (ei, ej)
            if e == e_x
                st.k_ins[x2, e] = 0.0
            else
                c   = (e == ej) ? cj : ci
                c_x = count_col(st.partition, e_x)
                st.k_ins[x2, e] = (st.ml[e, x2] + st.mc[e, c + 1] + st.mc[e_x, c_x - 1])
                                - (st.ml[e_x, x2] + st.mc[e, c] + st.mc[e_x, c_x])
            end
        end
        # note: the moved detection's own row is handled below, where its full row is stale
    end
    refresh_k_ins_row!(st, x)            # row x: element identity changed, all ne entries stale
    return nothing
end

"Recompute k_swp entries for all pairs involving detection x. O(nx); no mc terms."
function refresh_k_swp_pairs!(st::RTWState, x::Int)::Nothing
    nx = size(st.partition, 1)
    ex = st.assigned[x]
    @inbounds for y = 1:nx
        if y == x
            continue
        end
        i = upper_index(x, y, nx)
        ey = st.assigned[y]
        if ey == ex
            st.k_swp[i] = 0.0            # same-element pair: no-op delta
        else
            st.k_swp[i] = (st.ml[ey, x] + st.ml[ex, y])
                        - (st.ml[ex, x] + st.ml[ey, y])
        end
    end
    return nothing
end

"Full rebuild of k_ins. Init and debug fallback only."
function refresh_k_ins_all!(st::RTWState)::Nothing
    ins_kernel!(st.k_ins, st.partition, st.ml, st.mc)
    nx, ne = size(st.k_ins)
    @inbounds for x = 1:nx
        st.k_ins[x, st.assigned[x]] = 0.0
        st.nk_ins[x] = logsumexp(view(st.k_ins, x, :))
    end
    return nothing
end

"Full rebuild of k_swp. Init and debug fallback only."
function refresh_k_swp_all!(st::RTWState)::Nothing
    swap_kernel!(st.k_swp, st.partition, st.ml)
    return nothing
end

#--------------------------------------------------------------------------------
# HELPERS
#--------------------------------------------------------------------------------

"Linear index into the upper triangle (row-major order: (1,2),(1,3),...,(nx-1,nx))."
function upper_index(a::Int, b::Int, nx::Int)::Int
    if a > b
        a, b = b, a
    end
    return (a - 1) * nx - (a - 1) * a ÷ 2 + (b - a)
end

"Inverse of upper_index: map a linear upper-triangle index back to (a, b)."
function upper_to_pair(k::Int, n::Int)::Tuple{Int, Int}
    # adapted from https://stackoverflow.com/a/68581180
    # find a such that i falls in a's block: block sizes nx-1, nx-2, ..., 1
    i = n - 1 - floor(Int,sqrt(-8*k + 4*n*(n-1) + 1)/2 - 0.5)
    j = k + i + ( (n-i+1)*(n-i) - n*(n-1) )÷2
    return i, j
end


"Number of elements in the upper triangle of an nXn matrix"
function upper_t_size(n::Int64)
    Int64(n * (n-1) / 2)
end

"Number of active entries in column e of the partition matrix."
function count_col(partition::BitMatrix, e::Int)::Int
    nx = size(partition, 1)
    c = 0
    @inbounds for x = 1:nx
        c += partition[x, e]
    end
    return c
end

function count_assocs(partition::BitMatrix, pad::Int64 = 1)
    nx,ne = size(partition)
    counts = fill(pad, ne)
    @inbounds for x = 1:nx
        for e = 1:ne
            if partition[x, e]
                counts[e] += 1
                break
            end
        end
    end
    return counts
end


"""
    bitmatrix_to_ntuple(pmat::BitMatrix)::NTuple

Converts an (N × ne) BitMatrix into a stack-allocated NTuple{N, UInt16} key.
"""
@inline function partition_to_tuple(partition::BitMatrix)::NTuple
    nx = size(partition, 1)
    # Returns NTuple{nx, UInt16}
    ntuple(x -> UInt16(unsafe_find_true(view(partition, x, :))), nx)
end
