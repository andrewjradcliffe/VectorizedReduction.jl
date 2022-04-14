#
# Date created: 2022-04-14
# Author: aradclif
#
#
############################################################################################
# Version to handle search along first dimension; eliminate once LoopVectorization is
# fixed. Almost assuredly, the varargs version would have terrible performance, thus
# I do not even bother to include it.

function vfindminmax1(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(f, T), Dᴮ′)
    C = similar(A, Int, Dᴮ′)
    if 1 ∈ dims
        Dᴮ′′ = (1, Dᴮ′...)
        newdims = ntuple(d -> dims[d] + 1, Val(M))
        _vfindminmax!(f, op, init, reshape(B, Dᴮ′′), reshape(C, Dᴮ′′), reshape(A, 1, Dᴬ...), newdims)
    else
        _vfindminmax!(f, op, init, B, C, A, dims)
    end
    return B, CartesianIndices(A)[C]
end

vfindminmax1(f, op, init, A, dims::Int) = vfindminmax1(f, op, init, A, (dims,))

vfindmax1(f::F, A, dims) where {F<:Function} = vfindminmax1(f, >, typemin, A, dims)
vfindmin1(f::F, A, dims) where {F<:Function} = vfindminmax1(f, <, typemax, A, dims)
vfindmax1(A::AbstractArray, dims) = vfindminmax1(identity, >, typemin, A, dims)
vfindmin1(A::AbstractArray, dims) = vfindminmax1(identity, <, typemax, A, dims)

vfindmax1(f::F, A) where {F<:Function} = vfindminmax1(f, >, typemin, A, :)
vfindmin1(f::F, A) where {F<:Function} = vfindminmax1(f, <, typemax, A, :)
# ::AbstractArray required in order for kwargs interface to work
vfindmax1(A::AbstractArray) = vfindmax1(identity, A)
vfindmin1(A::AbstractArray) = vfindmin1(identity, A)

vfindmax1(f, A; dims=:) = vfindmax1(f, A, dims)
vfindmax1(A; dims=:) = vfindmax1(identity, A, dims)
vfindmin1(f, A; dims=:) = vfindmin1(f, A, dims)
vfindmin1(A; dims=:) = vfindmin1(identity, A, dims)


############################################################################################
function vtfindminmax1(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(f, T), Dᴮ′)
    C = similar(A, Int, Dᴮ′)
    Dᴮ′′ = (1, Dᴮ′...)
    newdims = ntuple(d -> dims[d] + 1, Val(M))
    _vtfindminmax!(f, op, init, reshape(B, Dᴮ′′), reshape(C, Dᴮ′′), reshape(A, 1, Dᴬ...), newdims)
    return B, CartesianIndices(A)[C]
end
vtfindminmax1(f, op, init, A, dims::Int) = vtfindminmax1(f, op, init, A, (dims,))

vtfindminmax1(f, op, init, A, dims::Int) = vtfindminmax1(f, op, init, A, (dims,))

vtfindmax1(f::F, A, dims) where {F<:Function} = vtfindminmax1(f, >, typemin, A, dims)
vtfindmin1(f::F, A, dims) where {F<:Function} = vtfindminmax1(f, <, typemax, A, dims)
vtfindmax1(A::AbstractArray, dims) = vtfindminmax1(identity, >, typemin, A, dims)
vtfindmin1(A::AbstractArray, dims) = vtfindminmax1(identity, <, typemax, A, dims)

vtfindmax1(f::F, A) where {F<:Function} = vtfindminmax1(f, >, typemin, A, :)
vtfindmin1(f::F, A) where {F<:Function} = vtfindminmax1(f, <, typemax, A, :)
# ::AbstractArray required in order for kwargs interface to work
vtfindmax1(A::AbstractArray) = vtfindmax1(identity, A)
vtfindmin1(A::AbstractArray) = vtfindmin1(identity, A)

vtfindmax1(f, A; dims=:) = vtfindmax1(f, A, dims)
vtfindmax1(A; dims=:) = vtfindmax1(identity, A, dims)
vtfindmin1(f, A; dims=:) = vtfindmin1(f, A, dims)
vtfindmin1(A; dims=:) = vtfindmin1(identity, A, dims)
