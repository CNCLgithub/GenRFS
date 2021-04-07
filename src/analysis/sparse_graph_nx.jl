export sparse_graph_nx

function sparse_graph_nx(es::RFSElements{T}, xs::Vector{T}) where {T}
    ls, table = associations(es, xs, fast = false)
    _, ls = normalize_weights(ls)
    nt = length(table)
    #g = MetaGraph(SimpleGraph(nt))
    g = nx.Graph()
    
    node_attributes = []
    p_graphs = []

    for (i, p) in enumerate(table)
        node_att = (i, Dict(["ls" => ls[i],
                             "p" => p]))
        push!(node_attributes, node_att)
        # pushing partition graphs to know the edges
        push!(p_graphs, partition_graph(p))
    end

    g.add_nodes_from(node_attributes)
    
    for (a,b) in combinations(1:nt, 2)
        pa = @>> p_graphs[a] edges collect
        pb = @>> p_graphs[b] edges collect

        diff = symdiff(pa, pb)
        dist = length(diff) * 0.5
        dist == 1.0 && g.add_edge(a, b)
    end

    return g
end

