module VectorizedReduction

using LoopVectorization, Static

export vvmapreduce, vvmapreduce!, vvreduce, vvsum, vvprod, vvmaximum, vvminimum, vvextrema,
    vtmapreduce, vtmapreduce!, vtreduce, vtsum, vtprod, vtmaximum, vtminimum, vtextrema

include("vmapreduce.jl")
include("vmapreduce_vararg.jl")
include("vmap.jl")

export vfindmin, vfindmax, vargmin, vargmax,
    vtfindmin, vtfindmax, vtargmin, vtargmax

include("vfindminmax.jl")
include("vfindminmax_vararg.jl")
include("vargminmax.jl")

export vfindmin1, vfindmax1, vargmin1, vargmax1,
    vtfindmin1, vtfindmax1, vtargmin1, vtargmax1
include("vfindminmax1.jl")
include("vargminmax1.jl")

export vlogsumexp, vtlogsumexp

include("vlogsumexp.jl")

export vlogsoftmax, vtlogsoftmax

include("vlogsoftmax.jl")

export vsoftmax, vtsoftmax

include("vsoftmax.jl")

end
