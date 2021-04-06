export walk_the_walk

abstract type AbstractWalk end

struct LogScoreStep
    logscore::Float64
    step::Int64
end

include("uniform_walk.jl")

function get_log_score(p::Vector{Vector{Int64}},
                       s_table, c_table)::Float64
    l = 0.0
    n_es, n_xs = size(s_table)
    for j in 1:n_es
        isinf(l) && break # no need to continue if impossible
        assoc = p[j]
        nassoc = length(assoc)
        l += c_table[j, nassoc + 1]
        isempty(assoc) && continue # support not valid if empty
        l += sum(s_table[j, assoc])
    end
    return l
end

function get_possible_moves(p::Vector{Vector{Int64}}, s_table, c_table)
    possible_moves = Vector{Vector{Vector{Int64}}}()
    n_es, n_xs = size(s_table)

    for x=1:n_xs
        del_p = deepcopy(p)
        e = @>> del_p findall(e -> x in e) first
        @>> del_p[e] filter!(i -> i != x)
        
        for j=1:n_es
            j == e && continue
            isinf(c_table[j, length(del_p[j])+1]) && continue
            new_p = deepcopy(del_p)
            push!(new_p[j], x)
            sort!(new_p[j])
            push!(possible_moves, new_p)
        end
    end
    
    return possible_moves
end

function walk_the_walk(walk::AbstractWalk, n_steps::Int64,
                       es::RFSElements{T}, xs::Vector{T};
                      ret_trajectory=false) where {T}
    
    # preparing the walk
    s_table = rfs_table(es, xs, support)
    c_table = rfs_table(es, collect(0:length(xs)), cardinality)
    println("SUPPORT")
    display(s_table)
    println("CARDINALITY")
    display(c_table)

    # initializing the first partition (all xs to the first element)
    cs = [length(xs); fill(0, length(es) - 1)]
    cur_p = partition_indeces(partition_push(cs, collect(1:length(xs)))[1])

    visited_partitions = Dict{Vector{Vector{Int64}}, LogScoreStep}()
    visited_partitions[cur_p] = LogScoreStep(get_log_score(cur_p, s_table, c_table), 0)
    
    trajectory = ret_trajectory ? Vector{Vector{Vector{Vector{Int64}}}}(undef, n_steps) : nothing

    for i=1:n_steps
        ps = get_possible_moves(cur_p, s_table, c_table)
        ls = @>> ps map(p -> get_log_score(p, s_table, c_table))
        
        if ret_trajectory
            trajectory[i] = ps
        end

        cur_p = make_step(walk, ps, ls, visited_partitions)
        
        @>> 1:length(ps) begin
            filter(j -> !(ps[j] in keys(visited_partitions)))
            foreach(j -> visited_partitions[ps[j]] = LogScoreStep(ls[j], j))
        end
    end
    
    return visited_partitions, trajectory
end
