#
# Date created: 2022-04-14
# Author: aradclif
#
#
############################################################################################
# Version to handle search along first dimension; eliminate once LoopVectorization is fixed.

vargminmax1(f::F, op::OP, init::I, A, dims) where {F, OP, I} =
    vfindminmax1(f, op, init, A, dims)[2]

vargmax1(f::F, A, dims) where {F<:Function} = vfindminmax1(f, >, typemin, A, dims)[2]
vargmin1(f::F, A, dims) where {F<:Function} = vfindminmax1(f, <, typemax, A, dims)[2]
vargmax1(A::AbstractArray, dims) = vargminmax1(identity, >, typemin, A, dims)
vargmin1(A::AbstractArray, dims) = vargminmax1(identity, <, typemax, A, dims)

# Over all dims
vargmax1(f::F, A) where {F<:Function} = vfindminmax1(f, >, typemin, A, :)[2]
vargmin1(f::F, A) where {F<:Function} = vfindminmax1(f, <, typemax, A, :)[2]
# ::AbstractArray required in order for kwargs interface to work
vargmax1(A::AbstractArray) = vargmax1(identity, A)
vargmin1(A::AbstractArray) = vargmin1(identity, A)

vargmax1(f, A; dims=:) = vargmax1(f, A, dims)
vargmax1(A; dims=:) = vargmax1(identity, A, dims)

vargmin1(f, A; dims=:) = vargmin1(f, A, dims)
vargmin1(A; dims=:) = vargmin1(identity, A, dims)

################
vtargminmax1(f::F, op::OP, init::I, A, dims) where {F, OP, I} =
    vtfindminmax1(f, op, init, A, dims)[2]

vtargmax1(f::F, A, dims) where {F<:Function} = vtfindminmax1(f, >, typemin, A, dims)[2]
vtargmin1(f::F, A, dims) where {F<:Function} = vtfindminmax1(f, <, typemax, A, dims)[2]
vtargmax1(A::AbstractArray, dims) = vtargminmax1(identity, >, typemin, A, dims)
vtargmin1(A::AbstractArray, dims) = vtargminmax1(identity, <, typemax, A, dims)

vtargmax1(f::F, A) where {F<:Function} = vtfindminmax1(f, >, typemin, A, :)[2]
vtargmin1(f::F, A) where {F<:Function} = vtfindminmax1(f, <, typemax, A, :)[2]
# ::AbstractArray required in order for kwargs interface to work
vtargmax1(A::AbstractArray) = vtargmax1(identity, A)
vtargmin1(A::AbstractArray) = vtargmin1(identity, A)

vtargmax1(f, A; dims=:) = vtargmax1(f, A, dims)
vtargmax1(A; dims=:) = vtargmax1(identity, A, dims)
vtargmin1(f, A; dims=:) = vtargmin1(f, A, dims)
vtargmin1(A; dims=:) = vtargmin1(identity, A, dims)
