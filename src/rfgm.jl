export RFGM

struct RFSTrace{T} <: Gen.Trace
    gen_fn::GenerativeFunction
    args::Tuple # elements
    choices::ChoiceMap
    retval::PersistentVector{T}
    score::Float64
    ptensor::BitArray{3}
    pscores::Vector{Float64}
end



@inline Gen.get_args(trace::RFSTrace) = trace.args
@inline Gen.get_retval(trace::RFSTrace) = trace.retval
@inline Gen.get_score(trace::RFSTrace) = trace.score
@inline Gen.get_choices(trace::RFSTrace) = trace.choices
@inline Gen.get_gen_fn(trace::RFSTrace) = trace.gen_fn

struct RFGM{T} <: GenerativeFunction{PersistentVector{T}, RFSTrace{T}}
    estimator::RFS{T}
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
        set_submap!(choices, i, xs[i])
    end
    weight = Gen.logpdf(gen_fn.estimator, xs, es, gen_fn.estimator_args...)
    (choices, weight, xs)
end

function RFSTrace(gen_fn::RFGM{T}, es, xs) where {T}
    pls, ptensor = associations(es, xs, gen_fn.estimator_args...)
    weight = logsumexp(pls)
    nx = length(xs)
    choices = choicemap()
    @inbounds for i = 1:nx
        set_submap!(choices, i, xs[i])
    end
    RFSTrace{T}(gen_fn, (es, ), choices, PersistentVector{T}(xs), weight, ptensor, pls)
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
    @assert contains(es, nx) "subset too small or too large for RFS"
    trace = RFSTrace(gen_fn, es, xs)
    (trace, trace.score)
end

# TODO
# function Gen.regenerate(gen_fn::RFGM{T}, args::Tuple, selection::Selection) where {T}
# end

mutable struct RFUpdateState
    new_atable::Matrix{Float64}
    new_ctable::Matrix{Float64}
    new_pls::Vector{Float64}
    prev_atable::Matrix{Float64}
    prev_ctable::Matrix{Float64}
    prev_pls::Vector{Float64}
    ptensor::Array{Bool, 3}
    to_revise::Vector{Int64}
end

function RFUpdateState(new_es, prev_es, xs, ptensor, prev_pls, to_revise)

    prev_ctable = rfs_table(prev_es, collect(0:length(xs)), cardinality)
    prev_atable = rfs_table(prev_es, xs, support)

    new_ctable = rfs_table(new_es, collect(0:length(xs)), cardinality)
    new_atable = rfs_table(new_es, xs, support)
    new_pls = Vector{Float64}(undef, length(prev_pls))


    RFUpdateState(new_atable, new_ctable, new_pls, prev_atable,
                  prev_ctable, prev_pls, ptensor, to_revise)
end

function process_retained!(gen_fn::RFGM{T}, args, argdiffs,
        state::RFUpdateState) where {T}

    nx, _, np = size(state.ptensor)
    @inbounds for j = 1:np
        weight = state.prev_pls[j]
        for (ei, e) = enumerate(state.to_revise)
            c = 0
            for xi = 1:nx
                state.ptensor[xi, e, j] || continue
                weight += (state.new_atable[xi, ei] -
                                state.prev_atable[xi, ei])
                c += 1
            end
            weight += (state.new_ctable[c, ei] -
                       state.prev_ctable[c, ei])
        end
        state.new_pls[j] = weight
    end

    return nothing
end

# CASE 1: new elements
# CASE 2: new observation; all or none?
#
function Gen.update(trace::RFSTrace{T}, args::Tuple, argdiffs::Tuple,
                    ::EmptyChoiceMap) where {T}
    gen_fn = get_gen_fn(trace)
    prev_es = get_args(trace)[1]
    ptensor = trace.ptensor
    prev_pls = trace.pscores
    xs = trace.retval
    new_es = args[1]
    ediffs = argdiffs[1]
    @assert ediffs.new_length == ediffs.prev_length "changed number of elements not currentl supported"

    if isempty(ediffs.updated)
        # TODO: return no change
        return trace
    end

    to_revise = collect(Int64, keys(ediffs.updated))

    new_es = new_es[to_revise]
    prev_es = prev_es[to_revise]
    state = RFUpdateState(new_es, prev_es, xs, ptensor, prev_pls,
                          to_revise)
    process_retained!(get_gen_fn(trace), args, argdiffs, state)
    new_trace = RFSTrace{T}(gen_fn, args, EmptyChoiceMap(),
                            xs, logsumexp(state.new_pls),
                            state.ptensor, state.new_pls)

    weight = new_trace.score - trace.score
    retdiff = NoChange()
    discard = choicemap()
    return (new_trace, weight, retdiff, discard)
end
