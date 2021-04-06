export walk_the_walk

abstract type AbstractWalk end

struct LogScoreStep
    logscore::Float64
    step::Int64
end

include("uniform_walk.jl")



function walk_the_walk(walk::AbstractWalk, n_steps::Int64,
                       es::RFSElements{T}, xs::Vector{T}) where {T}
    
    # preparing the walk
    s_table = rfs_table(es, xs, support)
    c_table = rfs_table(es, collect(0:length(xs)), cardinality)

    # initializing the first partition 
    cs = [length(xs); fill(0, length(es) - 1)]
    
    cur_p = partition_indeces(partition_push(cs, collect(1:length(xs)))[1])


    visited_partitions = Dict{Vector{Vector{Int64}}, LogScoreStep}()

    println(cur_p)     

    for i=1:n_steps

    end

end
