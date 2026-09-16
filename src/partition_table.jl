struct PartitionTable
    keys   :: Vector{PartitionKey}   # insertion order from the walk
    values :: Vector{Float64}        # parallel; per-trace mutable
end

Base.length(t::PartitionTable) = length(t.values)
Base.eltype(::Type{PartitionTable}) = Pair{PartitionKey, Float64}
Base.isempty(t::PartitionTable) = isempty(t.values)
Base.keys(t::PartitionTable) = t.keys
Base.values(t::PartitionTable) = t.values
Base.haskey(t::PartitionTable, key::PartitionKey) = key in t.keys
Base.getindex(t::PartitionTable, key::PartitionKey) =
    t.values[findfirst(==(key), t.keys)]
Base.get(t::PartitionTable, key::PartitionKey, default) = begin
    i = findfirst(==(key), t.keys)
    i === nothing ? default : t.values[i]
end

# --- iteration: yield (key, value) pairs, Dict-like ---
Base.iterate(t::PartitionTable, i::Int = 1) =
    i > length(t.values) ? nothing :
    (Pair(t.keys[i], t.values[i]), i + 1)

function PartitionTable(visited::Dict{PartitionKey, Float64})
    ks = Vector{PartitionKey}(undef, length(visited))
    vs = Vector{Float64}(undef, length(visited))
    for (i, (k, v)) in enumerate(visited)   # dict iteration order is fine
        ks[i] = k; vs[i] = v
    end
    PartitionTable(ks, vs)
end

function retained_copy(p::PartitionTable)::PartitionTable
    PartitionTable(p.keys, copy(p.values))
end
