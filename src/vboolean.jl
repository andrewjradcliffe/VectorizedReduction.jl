#
# Date created: 2022-03-25
# Author: aradclif
#
#
############################################################################################

# A surprising convenience opportunity -- albeit, a specific implementation will
# be necessary in order to utilize Array{Bool} rather than the default, which
# will get promoted to Array{Int}. Oddly, only works on Array{<:Integer} inputs.
# A workaround (inefficient) would be to sum Bools, then compare to length
# vany(f::F, A, dims) where {F} = vvmapreduce(f, |, zero, A, dims)
# vall(f::F, A, dims) where {F} = vvmapreduce(f, &, one, A, dims)
# vany(A, dims) = vvmapreduce(identity, |, zero, A, dims)
# vall(A, dims) = vvmapreduce(identity, &, one, A, dims)

# A curious solution: initialize with a value, rather than `zero` or `one`.
# Most likely, this is due promotion by LoopVectorization, which in this case
# breaks the ability to use anything but Array{<:Integer} for a Boolean predicate
# function. However, initializing to a Bool solves this, preventing any
# problems with the type conversion.
# Alas! Another issue which also happens to come up with findminmax, argminmax:
# Reductions on the first dimension are not well-defined, hence, the same
# errors occur. Awkwardly, the same fix could be used: re-shape to shift
# everything to +1 dimension, reduce, then drop the dimension.
# Unfortunately, this results in worse performance than just calling
# `any` from Julia Base. It would probably be faster to do vcount
# and then compare...
function vany(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Bool, Dᴮ′)
    _vvmapreduce_init!(f, |, false, B, A, dims)
    return B
end
function vall(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Bool, Dᴮ′)
    _vvmapreduce_init!(f, &, true, B, A, dims)
    return B
end
function vcount(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Int, Dᴮ′)
    _vvmapreduce!(f, +, zero, B, A, dims)
    return B
end
function vcount(f::F, A::AbstractArray{T, N}, ::Colon) where {F, T, N}
    ξ = 0
    @turbo for i ∈ eachindex(A)
        ξ += f(A[i])
    end
    return ξ
end
vcount(f::F, A) where {F} = vcount(f, A, :)

vcount(A::AbstractArray{Bool, N}, dims) where {N} = vcount(identity, A, dims)
vcount(A::AbstractArray{Bool, N}) where {N} = vcount(identity, A)


vany(A, dims) = vany(identity, A, dims)
vall(A, dims) = vall(identity, A, dims)

# Realistically, I would not recommend these, as they
# are inevitably slower than code that can break out of the loop.
# The only scenario I envision them being faster is when one has reason
# to believe that a fair portion of the elements would actually need
# to be checked in order to exit. If one suspects that the any/all
# is likely to short-circuit, then I recommend against vany/vall.
function vany(f::F, A::AbstractArray{T, N}, ::Colon) where {F, T, N}
    # # The proper implementation, but, alas, zero_mask problems in LoopVectorization
    # ξ = false
    # @turbo for i ∈ eachindex(A)
    #     ξ |= f(A[i])
    # end
    # return ξ
    # The not-so-great solution
    return vcount(f, A) != 0
end
vany(f::F, A) where {F<:Function} = vany(f, A, :)
vany(A::AbstractArray{Bool, N}) where {N} = vany(identity, A)
function vall(f::F, A::AbstractArray{T, N}, ::Colon) where {F, T, N}
    ξ = true
    @turbo for i ∈ eachindex(A)
        ξ &= f(A[i])
    end
    return ξ
end
vall(f::F, A) where {F<:Function} = vall(f, A, :)
vall(A::AbstractArray{Bool, N}) where {N} = vall(identity, A)

# function vany2(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
#     if 1 ∈ dims
#         newdims = ntuple(d -> dims[d] + 1, Val(M))
#         Dᴬ′ = ntuple(d -> d == 1 ? 1 : size(A, d - 1), Val(N+1))
#         B = vany(f, reshape(A, Dᴬ′), newdims)
#         return dropdims(B, dims=1)
#     else
#         return vany(f, A, dims)
#     end
# end

# A silly definition, but it is nonetheless possible... and, an improvement by 2x
function vmask(f::F, A::AbstractArray{T, N}) where {F, T, N}
    B = similar(A, Bool)
    @turbo for i ∈ eachindex(A)
        B[i] = f(A[i])
    end
    return B
end
vfindall(f::F, A::AbstractArray{T, N}) where {F, T, N} = CartesianIndices(A)[vmask(f, A)]
