module VectorizedReduction

using LoopVectorization, Static
include("utils.jl")


export vvmapreduce, vvmapreduce!, vvreduce, vvsum, vvprod, vvmaximum, vvminimum,
    vtmapreduce, vtmapreduce!, vtreduce, vtsum, vtprod, vtmaximum, vtminimum

include("vmapreduce.jl")
include("vmapreduce_vararg.jl")
include("vmap.jl")

export vvextrema, vtextrema
include("vextrema.jl")

export vfindmin, vfindmax, vargmin, vargmax,
    vtfindmin, vtfindmax, vtargmin, vtargmax

include("vfindminmax.jl")
include("vfindminmax_vararg.jl")
include("vargminmax.jl")

export vfindmin1, vfindmax1, vargmin1, vargmax1,
    vtfindmin1, vtfindmax1, vtargmin1, vtargmax1
include("vfindminmax1.jl")
include("vargminmax1.jl")

export vcount, vtcount, vany, vall, vtany, vtall
include("vboolean.jl")

export vlogsumexp, vtlogsumexp

include("vlogsumexp.jl")

export vlogsoftmax, vtlogsoftmax

include("vlogsoftmax.jl")

export vsoftmax, vtsoftmax

include("vsoftmax.jl")

export vmapreducethen, vtmapreducethen
include("vmapreducethen.jl")
include("vmapreducethen_vararg.jl")

export vnorm, vtnorm

include("vnorm.jl")

export vminkowski, vtminkowski,
    vmanhattan, vtmanhattan,
    veuclidean, vteuclidean,
    vchebyshev, vtchebyshev

include("vdistance.jl")

export vmean, vtmean,
    vgeomean, vtgeomean,
    vharmmean, vtharmmean,
    vmean_log, vtmean_log

export vlse, vtlse,
    vlse_mean, vtlse_mean

export ventropy, vtentropy,
    vcrossentropy, vtcrossentropy,
    vmaxentropy, vtmaxentropy,
    vshannonentropy, vtshannonentropy,
    vcollisionentropy, vtcollisionentropy,
    vminentropy, vtminentropy,
    vrenyientropy, vtrenyientropy

export vkldivergence, vtkldivergence,
    vgkldiv, vtgkldiv,
    vrenyadivergence, vtrenyadivergence

export vcounteq, vtcounteq,
    vcountne, vtcountne,
    vmeanad, vtmeanad,
    vmaxad, vtmaxad,
    vmse, vtmse,
    vrmse, vtrmse,
    vmsd, vtmsd,
    vrmsd, vtrmsd

include("vstats.jl")

end
