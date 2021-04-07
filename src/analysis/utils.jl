using LightGraphs

function partition_graph(p)

    e = length(p)
    x = @>> p filter(!isempty) map(maximum) maximum
    g = SimpleGraph(e + x)
    for (i, j) in enumerate(p)
        for k in j
            add_edge!(g, i, x + k)
        end
    end
    return g
end
