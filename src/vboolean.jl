#
# Date created: 2022-03-25
# Author: aradclif
#
#
############################################################################################

"""
    vcount([f=identity,] A::AbstractArray, dims=:)

Count the number of elements in `A` for which `f` return true over the given `dims`.
If `f` is omitted, count the number of `true` elements in `A`
(which should be `AbstractArray{Bool}`).
"""
function vcount(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Int, Dᴮ′)
    _vvmapreduce!(f, +, zero, B, A, dims)
    return B
end
vcount(f, A, dims::Int) = vcount(f, A, (dims,))
function vcount(f::F, A::AbstractArray{T, N}, ::Colon) where {F, T, N}
    ξ = 0
    @turbo for i ∈ eachindex(A)
        ξ += f(A[i])
    end
    return ξ
end
vcount(f::F, A::AbstractArray) where {F} = vcount(f, A, :)
vcount(A::AbstractArray{Bool, N}, dims) where {N} = vcount(identity, A, dims)
vcount(A::AbstractArray{Bool, N}) where {N} = vcount(identity, A)

# kwargs interface
"""
    vcount([f=identity,] A::AbstractArray; dims=:)

Identical to non-keyword args version; slightly less performant due to use of kwargs.
"""
vcount(f, A; dims=:) = vcount(f, A, dims)
vcount(A; dims=:) = vcount(A, dims)

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
"""
    vany([p=identity,] A::AbstractArray, dims=:)

Determine whether predicate `p` returns true for any elements over the given `dims`.
If `p` is omitted, test whether any values along the given `dims` are `true`
(in which case `A` should be `AbstractArray{Bool}`).

# Usage Recommendation
If `A` is reasonably small, `vany` may be faster than `any`; however, as the size
of `A` grows, the probability of any element returning true inevitably increases
(it is repeated Bernoulli sampling, thus, even with a very small success probability,
 a large number of tries makes may yield a scenario where the `break` of `any` wins out).
Consequently, the probability of individual elements being `true` should determine choice --
if one suspects a reasonable success probability, then `any` may be preferable, depending
on the size `A`. More testing is needed to determine potential breakpoints.

# Additional Notes
This function suffers from the same issue as `vfindmax` and friends -- reductions
which include the first dimension with zero masks are not yet supported by LoopVectorization.
Notably, this function still works as intended for any reduction which does not involve
the first dimension.
"""
function vany(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Bool, Dᴮ′)
    _vvmapreduce_init!(f, |, false, B, A, dims)
    return B
end
vany(f, A, dims::Int) = vany(f, A, (dims,))

"""
    vall([p=identity,] A::AbstractArray, dims=:)

Determine whether predicate `p` returns true for all elements over the given `dims`.
If `p` is omitted, test whether all values along the given `dims` are `true`
(in which case `A` should be `AbstractArray{Bool}`).

# Additional Notes
This function suffers from the same issue as `vfindmax` and friends -- reductions
which include the first dimension with max masks are not yet supported by LoopVectorization.
Notably, this function still works as intended for any reduction which does not involve
the first dimension.
"""
function vall(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Bool, Dᴮ′)
    _vvmapreduce_init!(f, &, true, B, A, dims)
    return B
end
vall(f, A, dims::Int) = vall(f, A, (dims,))

vany(A::AbstractArray, dims) = vany(identity, A, dims)
vall(A::AbstractArray, dims) = vall(identity, A, dims)

# kwargs interface
"""
    vany([p=identity,] A::AbstractArray; dims=:)

Identical to non-keyword args version; slightly less performant due to use of kwargs.
"""
vany(f, A; dims=:) = vany(f, A, dims)
vany(A; dims=:) = vany(identity, A, dims)

"""
    vall([p=identity,] A::AbstractArray; dims=:)

Identical to non-keyword args version; slightly less performant due to use of kwargs.
"""
vall(f, A; dims=:) = vall(f, A, dims)
vall(A; dims=:) = vall(identity, A, dims)

# Realistically, I would not recommend these, as they
# are inevitably slower than code that can break out of the loop.
# The only scenario I envision them being faster is when one has reason
# to believe that a fair portion of the elements would actually need
# to be checked in order to exit. If one suspects that the any/all
# is likely to short-circuit, then I recommend against vany/vall.
function vany(f::F, A::AbstractArray{T, N}, ::Colon) where {F, T, N}
    # The proper implementation, but, alas, zero_mask problems in LoopVectorization
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
# Note: an improvement only when A is reasonably small, perhaps 10^4 elements at most,
# at which point the cost of CartesianIndices(A)[B] becomes the limiting factor.
function vmask(f::F, A::AbstractArray{T, N}) where {F, T, N}
    B = similar(A, Bool)
    @turbo for i ∈ eachindex(A)
        B[i] = f(A[i])
    end
    return B
end
vfindall(f::F, A::AbstractArray{T, N}) where {F, T, N} = CartesianIndices(A)[vmask(f, A)]
vfindall(f::F, A::AbstractVector{T}) where {F, T} = LinearIndices(A)[vmask(f, A)]

############################################################################################

"""
    vtcount([f=identity,] A::AbstractArray, dims=:)

Count the number of elements in `A` for which `f` return true over the given `dims`.
If `f` is omitted, count the number of `true` elements in `A`
(which should be `AbstractArray{Bool}`). Threaded.
"""
function vtcount(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Int, Dᴮ′)
    _vtmapreduce!(f, +, zero, B, A, dims)
    return B
end
vtcount(f, A, dims::Int) = vtcount(f, A, (dims,))
function vtcount(f::F, A::AbstractArray{T, N}, ::Colon) where {F, T, N}
    ξ = 0
    @tturbo for i ∈ eachindex(A)
        ξ += f(A[i])
    end
    return ξ
end
vtcount(f::F, A::AbstractArray) where {F} = vtcount(f, A, :)
vtcount(A::AbstractArray{Bool, N}, dims) where {N} = vtcount(identity, A, dims)
vtcount(A::AbstractArray{Bool, N}) where {N} = vtcount(identity, A)

# kwargs interface
"""
    vtcount([f=identity,] A::AbstractArray; dims=:)

Identical to non-keyword args version; slightly less performant due to use of kwargs. Threaded.
"""
vtcount(f, A; dims=:) = vtcount(f, A, dims)
vtcount(A; dims=:) = vtcount(A, dims)

"""
    vtany([p=identity,] A::AbstractArray, dims=:)

Determine whether predicate `p` returns true for any elements over the given `dims`.
If `p` is omitted, test whether any values along the given `dims` are `true`
(in which case `A` should be `AbstractArray{Bool}`). Threaded.

# Additional Notes
This function suffers from the same issue as `vfindmax` and friends -- reductions
which include the first dimension with zero masks are not yet supported by LoopVectorization.
Notably, this function still works as intended for any reduction which does not involve
the first dimension.
"""
function vtany(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Bool, Dᴮ′)
    _vtmapreduce_init!(f, |, false, B, A, dims)
    return B
end
vtany(f, A, dims::Int) = vtany(f, A, (dims,))

"""
    vtall([p=identity,] A::AbstractArray, dims=:)

Determine whether predicate `p` returns true for all elements over the given `dims`.
If `p` is omitted, test whether all values along the given `dims` are `true`
(in which case `A` should be `AbstractArray{Bool}`). Threaded.

# Additional Notes
This function suffers from the same issue as `vfindmax` and friends -- reductions
which include the first dimension with max masks are not yet supported by LoopVectorization.
Notably, this function still works as intended for any reduction which does not involve
the first dimension.
"""
function vtall(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Bool, Dᴮ′)
    _vtmapreduce_init!(f, &, true, B, A, dims)
    return B
end
vtall(f, A, dims::Int) = vtall(f, A, (dims,))

vtany(A::AbstractArray, dims) = vtany(identity, A, dims)
vtall(A::AbstractArray, dims) = vtall(identity, A, dims)

# kwargs interface
"""
    vtany([p=identity,] A::AbstractArray; dims=:)

Identical to non-keyword args version; slightly less performant due to use of kwargs.
"""
vtany(f, A; dims=:) = vtany(f, A, dims)
vtany(A; dims=:) = vtany(identity, A, dims)

"""
    vtall([p=identity,] A::AbstractArray; dims=:)

Identical to non-keyword args version; slightly less performant due to use of kwargs.
"""
vtall(f, A; dims=:) = vtall(f, A, dims)
vtall(A; dims=:) = vtall(identity, A, dims)

function vtany(f::F, A::AbstractArray{T, N}, ::Colon) where {F, T, N}
    return vtcount(f, A) != 0
end
vtany(f::F, A) where {F<:Function} = vtany(f, A, :)
vtany(A::AbstractArray{Bool, N}) where {N} = vtany(identity, A)
function vtall(f::F, A::AbstractArray{T, N}, ::Colon) where {F, T, N}
    ξ = true
    @tturbo for i ∈ eachindex(A)
        ξ &= f(A[i])
    end
    return ξ
end
vtall(f::F, A) where {F<:Function} = vtall(f, A, :)
vtall(A::AbstractArray{Bool, N}) where {N} = vtall(identity, A)

function vtmask(f::F, A::AbstractArray{T, N}) where {F, T, N}
    B = similar(A, Bool)
    @tturbo for i ∈ eachindex(A)
        B[i] = f(A[i])
    end
    return B
end
vtfindall(f::F, A::AbstractArray{T, N}) where {F, T, N} = CartesianIndices(A)[vtmask(f, A)]
vtfindall(f::F, A::AbstractVector{T}) where {F, T} = LinearIndices(A)[vtmask(f, A)]
