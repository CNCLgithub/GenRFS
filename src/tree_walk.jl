using LinearAlgebra
# using DataStructures # TODO: Remove

const MAX_XS = 64

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

const KINS_MISMATCH = Ref(0)

function check_kernels(st::RTWState, tag::String)
    k_ref = ins_kernel(st.partition, st.ml, st.mc)
    ks_ref = swap_kernel(st.partition, st.ml)
    bad_ins = findall(!isapprox(k_ref[i], st.k_ins[i]; atol = 1e-8) for i in eachindex(k_ref))
    bad_swp = findall(!isapprox(ks_ref[i], st.k_swp[i]; atol = 1e-8) for i in eachindex(ks_ref))
    if !isempty(bad_ins) || !isempty(bad_swp)
        KINS_MISMATCH[] += 1
        if KINS_MISMATCH[] <= 3   # print first 3 occurrences only
            @warn "kernel drift after $tag" n_ins_bad = length(bad_ins) n_swp_bad = length(bad_swp)
            for i in first(bad_ins, 3)
                x = Int(ceil(i / size(k_ref, 2)))   # column-major
                e = i - (x - 1) * size(k_ref, 2)
                @info "k_ins[$x,$e] refreshed=$(st.k_ins[x,e]) exact=$(k_ref[x,e]) assigned=$(st.assigned[x]) count=$(count_idx(st.partition, e))"
            end
        end
    end
end

function mcmc_tree_step_debug!(st::RTWState, t::Float64 = 1.0, p_swap::Float64 = 0.5)
    if rand() < p_swap
        metro_swap!(st, t)
        check_kernels(st, "swap")
    else
        gibbs_insert!(st, t)
        check_kernels(st, "insert")
    end
end


# function mcmc_tree_step_debug!(st::RTWState, t::Float64=1.0, p_swap::Float64=0.5)
#     mcmc_tree_step!(st, t, p_swap)
#     fresh = partition_score(st.partition, st.ml, st.mc)
#     if !isapprox(st.pscore, fresh; atol=1e-8)
#         @warn "pscore drift" st.pscore fresh (fresh - st.pscore)
#     end
#     # also: full kernel consistency
#     k_ref = ins_kernel(st.partition, st.ml, st.mc)
#     @assert isapprox(k_ref, st.k_ins; atol=1e-8) "k_ins drift"
#     ks_ref = swap_kernel(st.partition, st.ml)
#     @assert isapprox(ks_ref, st.k_swp; atol=1e-8) "k_swp drift"
# end


"Composed MCMC step: swaps w.p. p_swap (ergodicity under finite-support cardinality), else Gibbs insert."
function mcmc_tree_step!(st::RTWState, t::Float64=1.0, p_swap::Float64=0.5)::Nothing
    if rand() < p_swap
        # MH move swapping ej -> ei
        metro_swap!(st, t)
    else
        # Gibbs move, reassigning x_i from e_j to e_k
        gibbs_insert!(st, t)
        # biased_insert!(st, t)
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
        st.rowbuf[j] = st.k_ins[x, j]            # RAW deltas, no exp
    end
    ej = unsafe_categorical!(st.rowbuf, t)       # sampler does temp + exp + normalize

    ej == 0 && error("Enable to random walk insert")
    ej == ei && return

    st.partition[x, ei] = false
    st.partition[x, ej] = true
    st.assigned[x] = ej
    st.pscore += st.k_ins[x, ej]

    refresh_k_ins_cols!(st, x, ei, ej)   # O(2*nx + ne)
    refresh_k_ins_row!(st, x) 
    refresh_k_swp_pairs!(st, x)          # O(nx): pairs involving x
    idx = partition_to_tuple(st.partition)
    # println("Score after insert: $(st.pscore)")
    haskey(st.visited, idx) || (st.visited[idx] = st.pscore)
    # update_after_insert!(st, x, ei, ej)
    return nothing
end

function biased_insert!(st::RTWState, t::Float64 = 1.0)::Nothing
    nx, ne = size(st.partition)
    x  = rand(1:nx)
    ei = st.assigned[x]
    ci = count_idx(st.partition, ei)

    # O(ne) proposal weights: numerator terms only, tempered
    @inbounds for e = 1:ne
        if e == ei
            st.rowbuf[e] = 0.0
            continue
        end
        cj  = count_idx(st.partition, e)
        delta = (st.ml[e, x] + st.mc[e, cj + 1] + st.mc[ei, ci - 1]) -
            (st.ml[ei, x] + st.mc[ei, ci]     + st.mc[e, cj])
        st.rowbuf[e] = delta
    end
    ej = unsafe_categorical!(st.rowbuf, 1.0)

    ej === ei || ej === 0 && return nothing   # stay: no state change

    cj  = count_idx(st.partition, ej)
    Δ   = (st.ml[ej, x] + st.mc[ej, cj + 1] + st.mc[ei, ci - 1]) -
          (st.ml[ei, x] + st.mc[ei, ci]     + st.mc[ej, cj])

    # partition_insert_move!(st, x, ei, ej)
    st.partition[x, ei] = false
    st.partition[x, ej] = true
    st.assigned[x] = ej
    st.pscore += Δ                              # exact, incremental

    key = partition_to_tuple(st.partition)
    haskey(st.visited, key) || (st.visited[key] = st.pscore)
    return nothing
end


"One Metropolis swap step: uniform proposal over all pairs (state-independent q)."
function metro_swap!(st::RTWState, t::Float64)::Nothing
    nx = size(st.partition, 1)
    # sample a random pair as an index into upper triangle
    N  = nx * (nx - 1) ÷ 2
    i  = rand(1:N)
    a, b = upper_to_pair(i, nx)
    ea, eb = st.assigned[a], st.assigned[b]
    eb == ea && return nothing           # a,b assigned to same element
    w = st.k_swp[i]                      # exact Δscore; move is self-inverse
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
    K = PartitionKey
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
    ne, nx = size(l_table)
    max_ls = vec(maximum(l_table, dims = 1))
    partition = zeros(Bool, (nx, ne))
    assigned = Vector{Int64}(undef, nx)
    # start with the "closest" assignment
    @inbounds @views for xi = sortperm(max_ls, rev = true)
        # prefer most restricted elements in terms of constraints
        for ei = sortperm(max_charges)
            count(partition[:, ei]) >= max_charges[ei] && continue
            partition[xi, ei] = true
            assigned[xi] = ei
            break
        end
    end
    (BitMatrix(partition), assigned)
end


function ins_kernel(partition::BitMatrix,
                    l_table::Matrix{Float64},
                    c_table::Matrix{Float64})
    (ne, nx) = size(l_table)
    k_ins = zeros((nx, ne))
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
                # 0 log weight transition to self
                0.0
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
        ei = unsafe_find_true(partition[a, :])
        laei = l_table[ei, a]
        for b = (a+1):nx
            i += 1
            ej = unsafe_find_true(partition[b, :])
            if ei == ej
                # can't swap when assigned to same element
                k_swap[i] = 0.0
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
    # println("Inserting $x from $ei to $ej, w=$(st.k_ins[x, ej])")
    # println("P-score before: $(st.pscore)")
    st.pscore += st.k_ins[x, ej]
    # println("P-score after: $(st.pscore)")
    return nothing
end

function partition_swap_move!(st::RTWState, a::Int, b::Int)::Nothing
    ea, eb = st.assigned[a], st.assigned[b]
    st.partition[a, ea] = false; st.partition[a, eb] = true
    st.partition[b, eb] = false; st.partition[b, ea] = true
    st.assigned[a], st.assigned[b] = eb, ea
    idx = upper_index(a, b, size(st.partition, 1))
    # println("Swapping $ei and $ej; w=$(st.k_swp[idx])")
    # println("P-score before: $(st.pscore)")
    st.pscore += st.k_swp[idx]
    # println("P-score after: $(st.pscore)")
    return nothing
end

#--------------------------------------------------------------------------------
# MOVE BOOKKEEPING
#--------------------------------------------------------------------------------

"Restore kernel consistency after insert move x: ei -> ej."
function update_after_insert!(st::RTWState, x::Int, ei::Int, ej::Int)::Nothing
    refresh_k_ins_cols!(st, x, ei, ej)   # O(2*nx + ne)
    refresh_k_ins_row!(st, x) 
    refresh_k_swp_pairs!(st, x)          # O(nx): pairs involving x
    idx = partition_to_tuple(st.partition)
    # println("Score after insert: $(st.pscore)")
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
    # println("Score after swap: $(st.pscore)")
    haskey(st.visited, idx) || (st.visited[idx] = st.pscore)
    return nothing
end

#--------------------------------------------------------------------------------
# KERNEL REFRESH PROCEDURES
#--------------------------------------------------------------------------------

"Recompute row x of k_ins. Called when x's element identity changes."
function refresh_k_ins_row!(st::RTWState, x::Int)::Nothing
    ne = size(st.k_ins, 2)
    ej = st.assigned[x]
    ej_idx = count_idx(st.partition, ej)
    @inbounds for e = 1:ne
        if e == ej
            st.k_ins[x, e] = 0.0
        else
            e_idx = count_idx(st.partition, e)
            st.k_ins[x, e] = ((st.ml[e, x] + st.mc[e, e_idx + 1] + st.mc[ej, ej_idx - 1])
                              - (st.ml[ej, x] + st.mc[ej, ej_idx] + st.mc[e, e_idx]))
        end
    end
    return nothing
end



"""
Recompute k_ins entries stale after insert move x: ei -> ej.
Members of ei/ej need their full row (their source count changed);
all other rows need only the ei/ej columns.
"""
function refresh_k_ins_cols!(st::RTWState, x::Int, ei::Int, ej::Int)::Nothing
    nx = size(st.k_ins, 1)
    ei_idx = count_idx(st.partition, ei)    # post-move table indices
    ej_idx = count_idx(st.partition, ej)
    @inbounds for x2 = 1:nx
        x2 == x && continue                 # handled by refresh_k_ins_row! below
        e_x = st.assigned[x2]
        if e_x == ei || e_x == ej
            refresh_k_ins_row!(st, x2)      # source count changed → full row
        else
            ex_idx = count_idx(st.partition, e_x)
            for e in (ei, ej)
                e_idx = (e == ej) ? ej_idx : ei_idx
                st.k_ins[x2, e] = (
                    (st.ml[e, x2] + st.mc[e, e_idx + 1] + st.mc[e_x, ex_idx - 1])
                    - (st.ml[e_x, x2] + st.mc[e_x, ex_idx] + st.mc[e, e_idx])
                )
            end
        end
    end
    refresh_k_ins_row!(st, x)               # row x: element identity changed, all ne entries stale
    return nothing
end

"Recompute k_swp entries for all pairs involving detection x"
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
        # st.nk_ins[x] = logsumexp(view(st.k_ins, x, :))
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

"""
    count_idx(partition, e) = count_col(partition, e) + 1

Table index for the *current* count of element e, per the cardinality_table
convention mc[e, n+1] = Pr(count = n). Post-move states use count_idx(…) ± 1.
"""
@inline count_idx(partition::BitMatrix, e::Int)::Int = count_col(partition, e) + 1

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

Converts an (N × ne) BitMatrix into a fixed-size key NTuple{MAX_XS, UInt16}.
Unassigned slots are 0 (valid partitions never assign to element 0).
One key type for the whole run: no recompilation per detection count.
"""

"Fixed-size partition key: one type for the whole run. Slots beyond `nx`
are 0 (valid partitions never assign to element 0). Using a fixed key type
avoids per-detection-count recompilation of RTWState, traces, and dicts."
const PartitionKey = NTuple{MAX_XS, UInt16}

"Pad a short assignment vector/tuple to a `PartitionKey`."
@inline function partition_key_from_vector(key::Vector{UInt16})::PartitionKey
    nx = length(key)
    @assert nx <= MAX_XS
    ntuple(Val(MAX_XS)) do i
        i <= nx ? key[i] : UInt16(0)
    end
end

@inline function partition_to_tuple(partition::BitMatrix)::PartitionKey
    nx = size(partition, 1)
    @assert nx <= MAX_XS
    # Padded to NTuple{MAX_XS, UInt16}; slots > nx are 0x0000
    ntuple(Val(MAX_XS)) do x
        x <= nx ? UInt16(unsafe_find_true(view(partition, x, :))) : UInt16(0)
    end
end
