using GenRFS, Gen, Random

using GenRFS: RTWState, ins_kernel, swap_kernel, support_table, cardinality_table,
    upper, max_assignment, partition_score, PartitionKey, partition_to_tuple,
    count_idx, metro_swap!, gibbs_insert!, upper_to_pair, upper_index, partition_insert_move!,
    update_after_insert!

using Printf

const REPORT_LIMIT = 3
const N_REPORTS = Ref(0)

# ---------------------------------------------------------------------------
# Ground truth: full kernel rebuild from current partition
# ---------------------------------------------------------------------------
kernels_exact(st::RTWState) =
    (ins_kernel(st.partition, st.ml, st.mc), swap_kernel(st.partition, st.ml))

same(a, b) = isequal(a, b) || isapprox(a, b; atol = 1e-8)

# ---------------------------------------------------------------------------
# Kernel consistency check (column-major-safe decoding)
# ---------------------------------------------------------------------------
function check_kernels(st::RTWState, tag::String)
    k_ins_exact, k_swp_exact = kernels_exact(st)
    nx, ne = size(st.k_ins)

    bad_ins = Tuple{Int,Int}[]
    for e in 1:ne, x in 1:nx
        same(st.k_ins[x, e], k_ins_exact[x, e]) || push!(bad_ins, (x, e))
    end

    bad_swp = Int[]
    for i in eachindex(st.k_swp)
        same(st.k_swp[i], k_swp_exact[i]) || push!(bad_swp, i)
    end

    (isempty(bad_ins) && isempty(bad_swp)) && return
    N_REPORTS[] += 1
    N_REPORTS[] > REPORT_LIMIT && return
    @warn "kernel drift after $tag" n_ins_bad=length(bad_ins) n_swp_bad=length(bad_swp) n_nan=sum(isnan, st.k_ins)
    for (x, e) in first(bad_ins, REPORT_LIMIT)
        @info "k_ins[$x,$e] refreshed=$(st.k_ins[x,e]) exact=$(k_ins_exact[x,e]) " *
              "assigned=$(st.assigned[x]) count_e=$(count_idx(st.partition, e)-1)"
    end
    for i in first(bad_swp, REPORT_LIMIT)
        a, b = upper_to_pair(i, nx)
        @info "k_swp[$i] refreshed=$(st.k_swp[i]) exact=$(k_swp_exact[i]) pair=($a,$b) " *
              "assigned=($(st.assigned[a]),$(st.assigned[b]))"
    end
end

# ---------------------------------------------------------------------------
# partition / assigned desync check
# ---------------------------------------------------------------------------
function check_state(st::RTWState)
    nx, ne = size(st.partition)
    for x in 1:nx
        n = sum(st.partition[x, :])
        n == 1 || @warn "detection $x has $n assignments; assigned=$(st.assigned[x])"
    end
    for e in 1:ne
        c = count_idx(st.partition, e) - 1
        c == sum(st.assigned .== e) ||
            @warn "count desync e=$e: partition=$c assigned=$(sum(st.assigned .== e))"
    end
end

# ---------------------------------------------------------------------------
# Instrumented move wrapper
# ---------------------------------------------------------------------------
function mcmc_tree_step_debug!(st::RTWState, t::Float64 = 1.0, p_swap::Float64 = 0.5)
    tag = rand() < p_swap ? "swap" : "insert"
    tag == "swap" ? metro_swap!(st, t) : gibbs_insert!(st, t)
    check_kernels(st, tag)
    check_state(st)
end

# --- Experiment B: rebuild one kernel at a time after each step -------------
# --- Experiment B: rebuild one kernel at a time after each step -------------
function mcmc_tree_step_bisect!(st::RTWState, t::Float64 = 1.0, p_swap::Float64 = 0.5)
    tag = rand() < p_swap ? "swap" : "insert"
    pre_assigned = copy(st.assigned)
    tag == "swap" ? metro_swap!(st, t) : gibbs_insert!(st, t)

    # 1. identity desync: partition[x, assigned[x]] must be the only true entry
    nx = size(st.partition, 1)
    for x in 1:nx
        true_col = findfirst(==(true), st.partition[x, :])
        if true_col != st.assigned[x]
            @warn "identity desync det $x: assigned=$(st.assigned[x]) but partition says $true_col (was $(pre_assigned[x]) before)"
        end
    end

    # 2. rebuild each kernel; report which one drifted, then repair
    k_ins_exact, k_swp_exact = kernels_exact(st)
    bad_ins = any(!same(st.k_ins[i], k_ins_exact[i]) for i in eachindex(st.k_ins))
    bad_swp = any(!same(st.k_swp[i], k_swp_exact[i]) for i in eachindex(st.k_swp))
    if bad_ins || bad_swp
        @warn "drift after $tag" bad_ins bad_swp
        if bad_ins && !bad_swp
            @info "-> isolated to k_ins refresh"
        elseif bad_swp && !bad_ins
            @info "-> isolated to k_swp refresh (upper_index convention?)"
        else
            @info "-> both stale; likely a missed refresh region"
        end
        st.k_ins .= k_ins_exact; st.k_swp .= k_swp_exact   # repair and continue
    end
end


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
function run_debug(es, xs, steps::Int = 200, t::Float64 = 1.0)
    Random.seed!(0)
    ml = support_table(es, xs)
    mc = cardinality_table(es, length(xs))
    us = Int64.(clamp.(upper.(es), 0, length(xs)))
    println("max_charges = ", us)      # <-- inspect: are Poisson charges capped?
    part, assigned = max_assignment(ml, mc, us)
    println("init assigned = ", assigned)   # <-- 0 = unassigned detections?
    println("init score    = ", partition_score(part, ml, mc))
    k_ins = ins_kernel(part, ml, mc)
    k_swp = swap_kernel(part, ml)
    st = RTWState(ml, mc, part, partition_score(part, ml, mc), k_swp, k_ins,
                  similar(k_swp), similar(k_ins), assigned,
                  Vector{Float64}(undef, length(es)),
                  Dict{PartitionKey, Float64}())
    st.visited[partition_to_tuple(part)] = st.pscore

    println("== init check ==")
    check_state(st)
    check_kernels(st, "init")

    println("== stepping ==")
    for i in 1:steps
        before = st.pscore
        # mcmc_tree_step_debug!(st, t)
        mcmc_tree_step_bisect!(st, t)
        fresh = partition_score(st.partition, st.ml, st.mc)
        isapprox(st.pscore, fresh; atol = 1e-8) ||
            (@warn "pscore drift at step $i" st.pscore fresh; break)
        before != st.pscore || continue
    end

    println("== summary ==")
    println("visited          = ", length(st.visited))
    println("best visited     = ", maximum(values(st.visited)))
    println("pscore           = ", st.pscore)
    println("fresh rescore    = ", partition_score(st.partition, st.ml, st.mc))
    return st
end

# ---------------------------------------------------------------------------
# Usage — the failing MRFS test case
# ---------------------------------------------------------------------------
es = RandomFiniteElement{Float64}[
    BernoulliElement{Float64}(0.8, normal, (0.0, 0.5)),
    BernoulliElement{Float64}(0.6, normal, (2.0, 0.5)),
    PoissonElement{Float64}(2.0, normal, (4.0, 0.5)),
]
xs = [0.1, 2.1, 3.9, 4.2]
st = run_debug(es, xs, 200, 1.0)
println("RFS reference    = ", Gen.logpdf(RFS{Float64}(), xs, es))
println("MRFS (plain)     = ", Gen.logpdf(MRFS{Float64}(), xs, es, 5000, 1.0))

# Case 1: large observation set (the failing one)
es = RandomFiniteElement{Float64}[
    PoissonElement{Float64}(3.0, normal, (0.0, 1.0)),
    PoissonElement{Float64}(2.0, normal, (5.0, 1.0)),
    BernoulliElement{Float64}(0.7, normal, (10.0, 0.5)),
]
xs = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 4.0, 4.5, 5.0, 5.5, 6.0, 10.1]
st = run_debug(es, xs, 5000, 1.0)
println("RFS reference    = ", Gen.logpdf(RFS{Float64}(), xs, es))
println("MRFS (plain)     = ", Gen.logpdf(MRFS{Float64}(), xs, es, 5000, 1.0))


# --- Experiment A: index-convention check (run once, before stepping) -------
function check_upper_conventions(nx::Int)
    ok_pair = all(upper_to_pair(upper_index(a, b, nx), nx) == (a, b)
                  for a in 1:(nx-1) for b in (a+1):nx)
    # rebuild-order check: swap_kernel! fills i in nested-loop order
    i = 0
    ok_order = true
    for a in 1:(nx-1), b in (a+1):nx
        i += 1
        upper_index(a, b, nx) == i || (ok_order = false; break)
    end
    @info "upper conventions" ok_pair ok_order
end
check_upper_conventions(13)
# ---- single-move forensics: force one insert and diff the refresh ----
using GenRFS, Random

Random.seed!(0)
es = RandomFiniteElement{Float64}[
    PoissonElement{Float64}(3.0, normal, (0.0, 1.0)),
    PoissonElement{Float64}(2.0, normal, (5.0, 1.0)),
    BernoulliElement{Float64}(0.7, normal, (10.0, 0.5)),
]
xs = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 4.0, 4.5, 5.0, 5.5, 6.0, 10.1]

ml = support_table(es, xs)
mc = cardinality_table(es, length(xs))
us = Int64.(clamp.(upper.(es), 0, length(xs)))
part, assigned = max_assignment(ml, mc, us)
k_ins = ins_kernel(part, ml, mc)
k_swp = swap_kernel(part, ml)
st = RTWState(ml, mc, part, partition_score(part, ml, mc), k_swp, k_ins,
              similar(k_swp), similar(k_ins), assigned,
              Vector{Float64}(undef, length(es)),
              Dict{PartitionKey, Float64}())

println("init consistent: ",
        all(isequal(st.k_ins[x,e], k_ins[x,e]) || isapprox(st.k_ins[x,e], k_ins[x,e]; atol=1e-9)
            for x in 1:size(k_ins,1), e in 1:size(k_ins,2)))

# pick a detection with a capacity-free alternative (det 13: elem 3 -> 1)
x, ej = 13, 1
ei = st.assigned[x]

pre_ins  = copy(st.k_ins)
pre_swp  = copy(st.k_swp)
delta    = st.k_ins[x, ej]

# gibbs_insert_move!(st, x, ej)   # or inline: partition_insert_move! + update_after_insert!
# If there is no exported single-move entry point, call:
partition_insert_move!(st, x, ei, ej); update_after_insert!(st, x, ei, ej)

post_ins = copy(st.k_ins)
post_swp = copy(st.k_swp)
exact_ins = ins_kernel(st.partition, st.ml, st.mc)
exact_swp = swap_kernel(st.partition, st.ml)

println("pscore: ", st.pscore, "  fresh: ", partition_score(st.partition, st.ml, st.mc))

for e in 1:size(st.k_ins, 2), xx in 1:size(st.k_ins, 1)
    a, b, c = post_ins[xx,e], exact_ins[xx,e], pre_ins[xx,e]
    if !isequal(a, b) && !isapprox(a, b; atol=1e-9)
        @printf("k_ins[%d,%d]  pre=%8.3f  post=%8.3f  exact=%8.3f  post-exact=%+8.3f  assigned=%d\n",
                xx, e, c, a, b, a - b, st.assigned[xx])
    end
end
for i in eachindex(post_swp)
    a, b, c = post_swp[i], exact_swp[i], pre_swp[i]
    if !isequal(a, b) && !isapprox(a, b; atol=1e-9)
        pa, pb = upper_to_pair(i, size(st.partition, 1))
        @printf("k_swp[%d](%d,%d)  pre=%8.3f  post=%8.3f  exact=%8.3f  post-exact=%+8.3f\n",
                i, pa, pb, c, a, b, a - b)
    end
end
println("st.assigned = ", st.assigned)
println("st.partition counts (per element) = ", [count(st.partition[:, e]) for e = 1:size(st.partition, 2)])
println("ml =")
display(round.(st.ml, digits=3))
println("mc =")
display(round.(st.mc, digits=3))
println("pre_ins (first 3 cols) =")
display(round.(pre_ins[:, 1:min(3, end)], digits=3))
