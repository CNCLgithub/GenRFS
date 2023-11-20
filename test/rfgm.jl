using Gen
using GenRFS


function test_call()
    gm = RFGM(RFS{Float64}(), ())
    be1 = BernoulliElement{Float64}(0.8, normal, (0., 1.0))
    be2 = BernoulliElement{Float64}(0.8, uniform, (-1, 1.0))
    es = RFSElements{Float64}([be1, be2])
    xs = gm(es)
    @show xs
    return nothing;
end


function test_propose()
    gm = RFGM(RFS{Float64}(), ())
    be1 = BernoulliElement{Float64}(0.8, normal, (0., 1.0))
    be2 = BernoulliElement{Float64}(0.8, uniform, (-1, 1.0))
    es = RFSElements{Float64}([be1, be2])
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
    es = RFSElements{Float64}([be1, be2])
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
    es = RFSElements{Float64}([be1, be2])
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
    es = RFSElements{Float64}([be1, be2])
    @time trace, _ = Gen.generate(gm, (es,), )

    be3 = BernoulliElement{Float64}(0.8, uniform, (-3, 3.0))
    new_es = RFSElements{Float64}([be1, be3])
    argdiffs = (Gen.VectorDiff(2, 2, Dict(2 => UnknownChange())),)
    @time new_trace, weight, _, _ = Gen.update(trace, (new_es,), argdiffs, EmptyChoiceMap())

    @show get_choices(new_trace)
    @show weight
    return nothing;
end


test_call();
test_propose();
test_simulate();
test_generate();
test_update();

