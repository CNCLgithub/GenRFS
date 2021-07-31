function prof_p_cube()
    a_table = randn(4, 8) .> 0
    us = fill(8, 4)
    b = @benchmark (GenRFS.partition_cube($a_table, $us))
    display(b)
    b = @benchmark (GenRFS.mem_partition_cube($a_table, $us))
    display(b)
end;

# prof_p_cube();

function prof_associations()
    GenRFS.modify_partition_ctx!(100)
    n = 5
    es = RFSElements{Float64}(undef, n)
    r = 4
    for i = 1:n
        es[i] = BernoulliElement{Float64}(0.5, uniform, (-1., 1.0))
    end
    es[n] = PoissonElement{Float64}(r, uniform, (-1., 1.0))

    xs = fill(0.1, 8)
    @time GenRFS.associations(es, xs)
    @time GenRFS.associations(es, xs)
    b = @benchmark GenRFS.associations($es, $xs)
    display(b)
    Profile.init(delay=1.0e-7,
                n = 10^7)
    @profilehtml GenRFS.associations(es, xs)
    # b = @benchmark (GenRFS.mem_partition_cube($a_table, $us))
    # display(b)
end;

prof_associations();
