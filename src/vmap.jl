#
# Date created: 2022-04-12
# Author: aradclif
#
#
############################################################################################
# A convenience, as map(f, A) and map(f, A...) are already defined
# However, there is no reason to use these in place of vmap and friends provided by C. Elrod
# as part of LoopVectorization.jl

function vvmap(f::F, A::AbstractArray{T, N}) where {F, T, N}
    B = similar(A, promote_type(Base.promote_op(f, T), Int))
    vvmap!(f, B, A)
    return B
end

@generated function vvmap!(f::F, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}) where {F, Tₒ, T, N}
    map_quote()
end

function vvmap(f::F, As::Tuple{Vararg{AbstractArray, P}}) where {F, P}
    # Actually, this check is unnecessary as `eachindex` will catch problems
    # ax = axes(As[1])
    # for p = 2:P
    #     axes(As[p]) == ax || throw(DimensionMismatch)
    # end
    B = similar(As[1], promote_type(Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int))
    vvmap!(f, B, As)
    return B
end

vvmap(f::F, As::Vararg{AbstractArray, P}) where {F, P} = vvmap(f, As)

@generated function vvmap!(f::F, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}) where {F, Tₒ, N, P}
    map_vararg_quote(P)
end


################
function vtmap(f::F, A::AbstractArray{T, N}) where {F, T, N}
    B = similar(A, promote_type(Base.promote_op(f, T), Int))
    vtmap!(f, B, A)
    return B
end

@generated function vtmap!(f::F, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}) where {F, Tₒ, T, N}
    tmap_quote()
end

function vtmap(f::F, As::Tuple{Vararg{AbstractArray, P}}) where {F, P}
    B = similar(As[1], promote_type(Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int))
    vtmap!(f, B, As)
    return B
end

vtmap(f::F, As::Vararg{AbstractArray, P}) where {F, P} = vtmap(f, As)

@generated function vtmap!(f::F, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}) where {F, Tₒ, N, P}
    tmap_vararg_quote(P)
end

