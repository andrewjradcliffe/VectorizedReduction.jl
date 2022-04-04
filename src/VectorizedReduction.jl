module VectorizedReduction

using LoopVectorization, Static

export vvmapreduce, vvmapreduce!, vvreduce, vvsum, vvprod, vvmaximum, vvminimum, vvextrema,
    vtmapreduce, vtmapreduce!, vtreduce, vtsum, vtprod, vtmaximum, vtminimum, vtextrema

include("vmapreduce.jl")

export vfindmin, vfindmax, vargmin, vargmax,
    vtfindmin, vtfindmax, vtargmin, vtargmax

include("vfindminmax.jl")
include("vargminmax.jl")

export vlogsumexp, vtlogsumexp

include("vlogsumexp.jl")

export vlogsoftmax, vtlogsoftmax

include("vlogsoftmax.jl")

export vsoftmax, vtsoftmax

include("vsoftmax.jl")

end
