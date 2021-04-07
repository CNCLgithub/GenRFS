using Gen
using GenRFS
#using GenRFS: sparse_graph
using LightGraphs
using EzXML
using GraphIO
using Lazy: @>>

p1 = PoissonElement{Float64}(3, normal, (0., 1.0))
b1 = BernoulliElement{Float64}(0.3, uniform, (-1.0, 1.0))
b2 = BernoulliElement{Float64}(0.3, uniform, (-1.0, 1.0))
b3 = BernoulliElement{Float64}(0.3, uniform, (-1.0, 1.0))
es = RFSElements{Float64}(undef, 4)
es[1] = p1
es[2] = b1
es[3] = b2
es[4] = b3

xs = [0.5, -0.5, 0.0, 0.]

n_steps = 50

visited_partitions, trajectory = walk_the_walk(UniformWalk(), n_steps, es, xs;
                                               ret_trajectory=true)

g = sparse_graph_nx(es, xs)

animate_walk(g, trajectory)

#savegraph("test.gexf", g, MyMLFormat())