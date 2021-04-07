export animate_walk

# returns Dict{partition => node in graph}
function get_part_to_node(g)
    part_to_node = Dict{Vector{Vector{Int64}}, Int64}()
    nodes = g.nodes.data()
    @>> nodes foreach(node -> part_to_node[node[2]["p"]] = node[1])
    return part_to_node
end

function get_node_to_ls(g)
    node_to_ls = Dict{Int64, Float64}()
    nodes = g.nodes.data()
    @>> nodes foreach(node -> node_to_ls[node[1]] = node[2]["ls"])
    return node_to_ls
end

function get_stable_attributes(g)
    pos = nx.spring_layout(g, iterations=1000)
    nodes = g.nodes.data()
    labels = @>> nodes map(n -> "$(n[2]["p"])")
    logscores = @>> nodes map(n -> n[2]["ls"])
    #logscores = @>> logscores map(ls -> max(ls, -1000000))

    return pos, labels, logscores
end


function update(i, g_ax, ls_ax, g, node_trajectory, nodes_visited, pos, labels, logscores, sum_ls)
    g_ax.clear()
    ls_ax.clear()

    g_ax.text(1, 1, "step $i")

    # basic graph
    nx.draw_networkx_edges(g, pos=pos, ax=g_ax, width=0.1, edge_color="gray")
    nx.draw_networkx_nodes(g, pos=pos, ax=g_ax, node_size=80, node_color=logscores, cmap=plt.cm.winter)
    
    nx.draw_networkx_nodes(g, pos=pos, ax=g_ax, node_size=30, nodelist=nodes_visited[i+1], node_color="purple")
    nx.draw_networkx_nodes(g, pos=pos, ax=g_ax, node_size=30, nodelist=node_trajectory[i+1], node_color="red")

    ls_ax.plot(sum_ls[1:i+1])
    ls_ax.set_xlim([1, length(node_trajectory)])
    ls_ax.set_ylim([0, 1])
    ls_ax.set_xlabel("step")
    ls_ax.set_ylabel("proportion of logscores")
end


function animate_walk(g, part_trajectory)
    
    # converting partition trajectory to node in g trajectory
    part_to_node = get_part_to_node(g)
    node_trajectory = @>> part_trajectory map(ps -> map(p -> part_to_node[p], ps))
    

    # finding the set 
    nodes_so_far = Vector{Int64}()
    nodes_visited = Vector{Vector{Int64}}(undef, length(node_trajectory))

    node_to_ls = get_node_to_ls(g)
    sum_ls = Vector{Float64}(undef, length(node_trajectory))

    for i=1:length(node_trajectory)
        nodes_so_far = union(nodes_so_far, node_trajectory[i])
        nodes_visited[i] = deepcopy(nodes_so_far)
        sum_ls[i] = @>> nodes_so_far map(n -> node_to_ls[n]) map(exp) sum
    end

    # initializing the figure
    fig, axs = plt.subplots(1,2)

    node_pos, node_labels, node_ls = get_stable_attributes(g)

    ani = animation.FuncAnimation(fig, update, frames=length(node_trajectory), interval=100, repeat=true,
                                  fargs=(axs[1], axs[2], g, node_trajectory, nodes_visited,
                                         node_pos, node_labels, node_ls,
                                         sum_ls))

    plt.show()
end
