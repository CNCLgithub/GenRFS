export UniformWalk

struct UniformWalk <: AbstractWalk end


function make_step(::UniformWalk, ps, ls, visited_partitions)
    ps[uniform_discrete(1, length(ps))]
end
