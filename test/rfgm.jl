using Gen
using GenRFS
using Profile
using StatProfilerHTML
using BenchmarkTools

const rfs_gm = RFGM(RFS{Float64}(), ())

@gen (static) function prior(i::Int64)
    mu = @trace(uniform(-10.0, 10.0), :mu)
    element::BernoulliElement{Float64} =
        BernoulliElement{Float64}(0.8, normal, (mu, 1.0))
    return element
end

map_prior = Gen.Map(prior)

@gen (static) function mygm(t::Int64, n::Int64)
    ids = fill(0, n)
    es = @trace(map_prior(ids), :es)
    xs = @trace(rfs_gm(es), :xs)
    return n
end

unfold_mygm = Gen.Unfold(mygm)

@gen static function myseqgm(t::Int64, k::Int64)
    steps = @trace(unfold_mygm(t, k), :t)
    return steps
end

function test_call()
    gm = RFGM(RFS{Float64}(), ())
    be1 = BernoulliElement{Float64}(0.8, normal, (0., 1.0))
    be2 = BernoulliElement{Float64}(0.8, uniform, (-1, 1.0))
    es = [be1, be2]
    xs = gm(es)
    @show xs
    return nothing;
end


function test_propose()
    gm = RFGM(RFS{Float64}(), ())
    be1 = BernoulliElement{Float64}(0.8, normal, (0., 1.0))
    be2 = BernoulliElement{Float64}(0.8, uniform, (-1, 1.0))
    es = [be1, be2]
    choices, weight, xs = Gen.propose(gm, (es,))
    @show choices
    @show weight
    @show xs
    return nothing;
end

function test_simulate()
    gm = RFGM(RFS{Float64}(), ())
    be1 = BernoulliElement{Float64}(0.8, normal, (0., 1.0))
    be2 = BernoulliElement{Float64}(0.8, uniform, (-1, 1.0))
    es = [be1, be2]
    trace = Gen.simulate(gm, (es,))
    @show get_choices(trace)
    @show get_score(trace)
    @show get_retval(trace)
    return nothing;
end


function test_generate()
    gm = RFGM(RFS{Float64}(), ())
    be1 = BernoulliElement{Float64}(0.8, normal, (0., 1.0))
    be2 = BernoulliElement{Float64}(0.8, uniform, (-1, 1.0))
    es = [be1, be2]
    (choices, _, _) = Gen.propose(gm, (es,))
    trace, weight = Gen.generate(gm, (es,), choices)
    @show get_choices(trace)
    @show weight
    return nothing;
end

function test_update()
    gm = RFGM(RFS{Float64}(), ())
    be1 = BernoulliElement{Float64}(0.8, normal, (0., 1.0))
    be2 = BernoulliElement{Float64}(0.8, uniform, (-1, 1.0))
    es = [be1, be2]
    @time trace = Gen.simulate(gm, (es,))

    be3 = BernoulliElement{Float64}(0.8, uniform, (-3, 3.0))
    new_es = [be1, be3]
    argdiffs = (Gen.VectorDiff(2, 2, Dict(2 => UnknownChange())),)
    @time new_trace, weight, _, _ = Gen.update(trace, (new_es,), argdiffs, EmptyChoiceMap())

    @show get_choices(new_trace)
    @show weight
    return nothing;
end

function _test_gm()
    for _ = 1:100
        Gen.simulate(mygm, (5,))
    end
end

function extract_rfs_subtraces(trace::Gen.Trace)
    t, _... = get_args(trace)
    # StaticIR names and nodes
    ir = Gen.get_ir(myseqgm)
    ir2 = Gen.get_ir(mygm)
    unfold_node = ir.call_nodes[1] # (:t,)
    unfold_fn = Gen.get_subtrace_fieldname(unfold_node)
    xs_node = ir2.call_nodes[2] # (:es, :xs)
    xs_fn = Gen.get_subtrace_fieldname(xs_node)
    # subtrace for each time step
    vector_trace = getproperty(trace, unfold_fn)
    result = Vector{GenRFS.RFSTrace{Float64}}(undef, t)
    for (idx, strace) in enumerate(vector_trace.subtraces)
        result[idx] = getproperty(strace, xs_fn)
    end
    return result
end

function test_gm()
    println("test gm")
    # trace, _ = Gen.generate(Gen.Map(prior), (collect(1:3),))
    # (new_trace, wdiff, retdiff) = Gen.regenerate(trace, select(1 => :mu))
    # display(get_choices(new_trace))
    # @show wdiff
    # @show retdiff
    #retdiff = VectorDiff(3, 3, Dict{Int64, Diff}(1 => UnknownChange()))

    # bm = @benchmark (Gen.generate($mygm, (5,)))
    # display(bm)

    # Profile.init(; n=10000000, delay=1E-6)
    # # @profilehtml _test_gm()
    # Profile.clear()
    # @profilehtml _test_gm()

    println("Generate")
    # bm = @benchmark (Gen.generate($myseqgm, (1,3)))
    # display(bm)
    trace, w = Gen.generate(myseqgm, (2,3))
    choices = get_choices(trace)
    # @show w
    # display(choices)
    args = (3, 3)
    argdiffs = (UnknownChange(), NoChange())
    # bm = @benchmark (Gen.update($trace, $args, $argdiffs,
    #                             choicemap((:t => 2 => :xs => 1, 0.),
    #                                       (:t => 2 => :xs => 2, 0.))))
    # display(bm)
    (new_trace, wdiff, retdiff, discard) =
        Gen.update(trace, args, argdiffs,
                   choicemap((:t => 3 => :xs => 1, 0.),
                             (:t => 3 => :xs => 2, 0.)))

    # @time (new_trace, wdiff, retdiff) =
    #     Gen.regenerate(trace, select(:es => 1 => :mu))
    # display(get_choices(new_trace))
    @show wdiff
    @show retdiff
    display(discard)

    bm = @benchmark (extract_rfs_subtraces($trace))
    display(bm)
    return nothing
end

test_call();
test_propose();
test_simulate();
test_generate();
test_update();
test_gm();

