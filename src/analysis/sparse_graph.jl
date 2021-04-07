using LightGraphs
using LightGraphs: AbstractGraphFormat
using MetaGraphs

import LightGraphs: savegraph
using EzXML

export MyMLFormat


function sparse_graph(es::RFSElements{T}, xs::Vector{T}) where {T}
    ls, table = associations(es, xs, fast = false)
    _, ls = normalize_weights(ls)
    nt = length(table)
    g = MetaGraph(SimpleGraph(nt))
    for (i, p) in enumerate(table)
        # @show i
        # @show p
        # @show ls[i]
        set_prop!(g, i, :ls, ls[i])
        set_prop!(g, i, :p, partition_graph(p))
        set_prop!(g, i, :Label, "$(p)")
    end
    for (a,b) in combinations(1:nt, 2)
        pa = @>> get_prop(g, a, :p) edges collect
        pb = @>> get_prop(g, b, :p) edges collect
        diff = symdiff(pa, pb)
        dist = length(diff) * 0.5
        dist == 1.0 && add_edge!(g, a, b)
    end
    return g
end

struct MyMLFormat <: AbstractGraphFormat end

mlformat(v) = "string"
mlformat(v::Int) = "int"
mlformat(v::Real) = "double"
mlformat(v::Bool) = "boolean"

function LightGraphs.savegraph(io::IO, g::AbstractMetaGraph, gname::String, ::MyMLFormat)
    xdoc = XMLDocument()
    xroot = setroot!(xdoc, ElementNode("graphml"))
    xroot["xmlns"] = "http://graphml.graphdrawing.org/xmlns"
    xroot["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
    xroot["xsi:schemaLocation"] = "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"

    xkeys = Dict{Symbol, String}()

    for (i, (k,v)) in enumerate(props(g, 1))
        key = "d$(i-1)"
        xkey = addelement!(xroot, "key")
        xkey["id"] = key
        xkey["for"] = "node"
        xkey["attr.name"] = k
        xkey["attr.type"] = mlformat(v)
        xkeys[k] = key
    end

    xg = addelement!(xroot, "graph")
    xg["id"] = 1
    xg["edgedefault"] = is_directed(g) ? "directed" : "undirected"



    for i in 1:nv(g)
        xv = addelement!(xg, "node")
        xv["id"] = "n$(i-1)"
        atts = props(g, i)
        for (att, v) in atts
            xvk = addelement!(xv, "data", "$(v)")
            xvk["key"] = xkeys[att]
        end
    end

    m = 0
    for e in LightGraphs.edges(g)
        xe = addelement!(xg, "edge")
        xe["id"] = "e$m"
        xe["source"] = "n$(src(e)-1)"
        xe["target"] = "n$(dst(e)-1)"
        m += 1
    end
    prettyprint(io, xdoc)
    return 1
end
