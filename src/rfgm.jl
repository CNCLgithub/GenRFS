export RFGM, RFSTrace

struct RFSTrace{T, K} <: Gen.Trace
    gen_fn::GenerativeFunction
    args::Tuple # elements
    choices::ChoiceMap
    retval::PersistentVector{T}
    score::Float64
    partitions::Dict{NTuple{K, UInt16}, Float64}
end

@inline Gen.get_args(trace::RFSTrace) = trace.args
@inline Gen.get_retval(trace::RFSTrace) = trace.retval
@inline Gen.get_score(trace::RFSTrace) = trace.score
@inline Gen.get_choices(trace::RFSTrace) = trace.choices
@inline Gen.get_gen_fn(trace::RFSTrace) = trace.gen_fn
@inline Gen.project(trace::RFSTrace, ::EmptySelection) = trace.score

struct RFGM{T} <: GenerativeFunction{PersistentVector{T}, RFSTrace{T}}
    estimator::AbstractRFS{T}
    estimator_args::Tuple
end

# TODO
Gen.has_argument_grads(gf::RFGM) = false
Gen.accepts_output_grad(gf::RFGM) = false

function (gen_fn::RFGM{T})(args...) where {T}
    es = args[1]
    PersistentVector{T}(sample_elements(es))
end

function Gen.propose(gen_fn::RFGM{T}, args::Tuple) where {T}
    es = args[1]
    xs = PersistentVector{T}(sample_elements(es))
    nx = length(xs)
    choices = choicemap()
    @inbounds for i = 1:nx
        choices[i] = xs[i]
        # set_submap!(choices, i, xs[i])
    end
    weight = Gen.logpdf(gen_fn.estimator, xs, es, gen_fn.estimator_args...)
    (choices, weight, xs)
end

function RFSTrace(gen_fn::RFGM{T}, es, xs) where {T}
    visited = associations(gen_fn.estimator, es, xs,
                           gen_fn.estimator_args...)
    weight = logsumexp_collection(values(visited))
    nx = length(xs)
    choices = choicemap()
    @inbounds for i = 1:nx
        choices[i] = xs[i]
    end
    K = length(xs)
    RFSTrace{T, K}(gen_fn, (es,), choices, PersistentVector{T}(xs),
                   weight, visited)
end

function Gen.simulate(gen_fn::RFGM{T}, args::Tuple) where {T}
    es = args[1]
    xs = sample_elements(es)
    RFSTrace(gen_fn, es, xs)
end

function Gen.generate(gen_fn::RFGM{T}, args::Tuple, ::EmptyChoiceMap) where {T}
    trace = simulate(gen_fn, args)
    (trace, trace.score)
end

function Gen.generate(gen_fn::RFGM{T}, args::Tuple, choices::ChoiceMap) where {T}
    es = args[1]
    xs = to_array(choices, T)
    nx = length(xs)
    if !contains(es, nx)
        error("Could not explain observed set $(xs) with elements $(es)")
    end
    trace = RFSTrace(gen_fn, es, xs)
    (trace, trace.score)
end

# TODO
# function Gen.regenerate(gen_fn::RFGM{T}, args::Tuple, selection::Selection) where {T}
# end

mutable struct RFUpdateState{K}
    new_atable::Matrix{Float64}
    new_ctable::Matrix{Float64}
    prev_atable::Matrix{Float64}
    prev_ctable::Matrix{Float64}
    partitions::Dict{NTuple{K, UInt16}, Float64}
    to_revise::Vector{Int64}
end

function RFUpdateState(new_es, prev_es, xs,
                       partitions::Dict{NTuple{K, UInt16}, Float64},
                       to_revise) where {K}
    nx = length(xs)
    prev_ctable = cardinality_table(prev_es, nx)
    prev_atable = support_table(prev_es, xs)
    new_ctable = cardinality_table(new_es, nx)
    new_atable = support_table(new_es, xs)
    RFUpdateState(new_atable, new_ctable, prev_atable, prev_ctable,
                  partitions, to_revise)
end

function process_retained!(state::RFUpdateState{K}) where {K}
    ne, nx = size(state.new_atable)
    to_revise = state.to_revise
    @inbounds for key in collect(keys(state.partitions))   # collect before mutating
        weight = state.partitions[key]
        for ei in to_revise
            c = 1                            # c = 1 denotes card-0, as before
            for xi = 1:nx
                key[xi] == ei || continue
                weight += (state.new_atable[ei, xi] -
                           state.prev_atable[ei, xi])
                c += 1
            end
            weight += (state.new_ctable[ei, c] -
                       state.prev_ctable[ei, c])
        end
        state.partitions[key] = weight
    end
    return nothing
end

# CASE 1: new elements
# CASE 2: new observation; all or none?

function compare_rfes(a, b)
    prev_length = length(a)
    new_length = length(b)
    diffs = Dict{Int64, Gen.Diff}()

    if prev_length === new_length
        for (ei, ea) = enumerate(a)
            if !in(ea, b)
                diffs[ei] = UnknownChange()
            end
        end
        if !isempty(diffs)
            return Gen.VectorDiff(prev_length, new_length, diffs)
        end

    else
        added = Set{RandomFiniteElement}(setdiff(b, a))
        deleted = Set{RandomFiniteElement}(setdiff(a, b))
        return SetDiff{RandomFiniteElement}(added, deleted)
    end

    return NoChange()
end

function Gen.update(trace::RFSTrace{T}, args::Tuple,
        argdiffs::Tuple{<:Gen.UnknownChange}, cm::ChoiceMap) where {T}
    prev_args = get_args(trace)
    diff = compare_rfes(prev_args[1], args[1])
    Gen.update(trace, args, (diff,), cm)
end

# TODO: update into new parent address? (e.g., Gen.Unfold)
# function Gen.update(trace::RFSTrace{T}, args::Tuple, argdiffs::Tuple{<:Gen.NoChange},
#                     ::ChoiceMap) where {T}
#     es = args[1]
#     xs = to_array(choices, T)
#     nx = length(xs)
#     @assert contains(es, nx) "subset too small or too large for RFS"
#     trace = RFSTrace(gen_fn, es, xs)
#     (trace, trace.score)
# end

function Gen.update(trace::RFSTrace{T}, args::Tuple, argdiffs::Tuple{<:D},
                    ::EmptyChoiceMap) where {T, D<:Gen.NoChange}
    (trace, 0.0, NoChange(), choicemap())
end

# REVIEW: should the ret-diff be `NoChange`?
function Gen.update(trace::RFSTrace{T}, args::Tuple, argdiffs::Tuple{<:Gen.SetDiff},
                    ::EmptyChoiceMap) where {T}
    gen_fn = get_gen_fn(trace)
    prev_es = get_args(trace)[1]
    xs = trace.retval
    new_es = args[1]

    # For now, just restart from scratch
    new_trace = RFSTrace(trace.gen_fn, new_es, xs)
    weight = new_trace.score - trace.score
    (new_trace, weight, NoChange(), choicemap())
end

function Gen.update(trace::RFSTrace{T, K}, args::Tuple, argdiffs::Tuple{<:Gen.VectorDiff},
                    ::EmptyChoiceMap) where {T, K}
    gen_fn = get_gen_fn(trace)
    prev_es = get_args(trace)[1]
    xs = trace.retval
    new_es = args[1]
    ediffs = argdiffs[1]

    @assert ediffs.new_length == ediffs.prev_length
    to_revise = collect(Int64, keys(ediffs.updated))
    state = RFUpdateState(new_es, prev_es, xs, trace.partitions, to_revise)
    process_retained!(state)
    new_trace = RFSTrace{T, K}(gen_fn, args, trace.choices,
                               xs, logsumexp_collection(values(state.partitions)),
                               state.partitions)
    weight = new_trace.score - trace.score
    retdiff = NoChange()
    discard = choicemap()
    return (new_trace, weight, retdiff, discard)
end

function Gen.regenerate(trace::GenRFS.RFSTrace{T}, args::Tuple,
        argdiffs::Tuple{UnknownChange}, selection::Selection) where {T}
    prev_args = get_args(trace)
    vdiff = compare_rfes(prev_args[1], args[1])
    Gen.regenerate(trace, args, (vdiff,), selection)
end

# REVIEW: What about the other selections?
function Gen.regenerate(trace::GenRFS.RFSTrace{T}, args::Tuple,
        argdiffs::Tuple{NoChange}, selection::EmptySelection) where {T}
    return (trace, 0.0, NoChange())
end

function Gen.regenerate(trace::GenRFS.RFSTrace{T}, args::Tuple,
        argdiffs::Tuple{<:Gen.SetDiff}, selection::EmptySelection) where {T}

    new_es = args[1]
    ediffs = argdiffs[1]
    xs = get_retval(trace)
    nret = length(xs)
    retdiff = (nochange() for _ = 1:nret)

    # For now, just restart from scratch
    new_trace = RFSTrace(trace.gen_fn, new_es, xs)
    weight = new_trace.score - trace.score
    (new_trace, weight, retdiff)
end

function Gen.regenerate(trace::GenRFS.RFSTrace{T}, args::Tuple,
        argdiffs::Tuple{<:Gen.VectorDiff}, selection::EmptySelection) where {T}

    new_es = args[1]
    ediffs = argdiffs[1]
    xs = get_retval(trace)
    nret = length(xs)
    retdiff = (nochange() for _ = 1:nret)


    @assert ediffs.new_length == ediffs.prev_length
    return process_elem_swap(trace, args, argdiffs)
end

function process_elem_swap(trace::GenRFS.RFSTrace{T,K}, args::Tuple,
                           argdiffs::Tuple{<:Gen.VectorDiff}) where {T,K}

    partitions = trace.partitions
    gen_fn = get_gen_fn(trace)
    prev_es = get_args(trace)[1]
    xs = trace.retval
    nret = length(xs)
    retdiff = (nochange() for _ = 1:nret)

    ediffs = first(argdiffs)
    new_es = args[1]

    to_revise = collect(Int64, keys(ediffs.updated))

    state = RFUpdateState(new_es, prev_es, xs, trace.partitions, to_revise)
    process_retained!(state)
    new_trace = RFSTrace{T, K}(gen_fn, args, trace.choices,
                               xs, logsumexp_collection(values(state.partitions)),
                               state.partitions)
    weight = new_trace.score - trace.score

    return (new_trace, weight, retdiff)
end

function Gen.project(trace::GenRFS.RFSTrace, selection::AllSelection)
    trace.score
end
