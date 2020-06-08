module GenRFS

using Gen
using Combinatorics

abstract type RFSParams end

include("utils.jl")
include("rfs/rfs.jl")

end # module
