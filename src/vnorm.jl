#
# Date created: 2022-09-13
# Author: aradclif
#
#
############################################################################################
_denom(A, dims) = prod(d -> size(A, d), dims)
_denom(A, ::Colon) = length(A)

# p-norms
_norm0(A, dims::NTuple{M, Int}) where {M} =
    fill!(similar(A, ntuple(d -> d ∈ dims ? 1 : size(A, d), Val(M))), _denom(A, dims))
_norm0(A, ::Colon) = float(length(A))

_vnorm(A, p::Rational{T}, dims) where {T} = _vnorm(A, float(p), dims)
Base.@constprop :aggressive function _vnorm(A, p, dims)
    if p == 2
        vmapreducethen(abs2, +, √, A, dims=dims)
    elseif p == 1
        vvsum(abs, A, dims=dims)
    elseif p == Inf
        vvmaximum(abs, A, dims=dims)
    elseif p == -Inf
        vvminimum(abs, A, dims=dims)
    else
        _vnorm_impl(A, p, dims)
    end
end
_vnorm_impl(A, p::T, dims) where {T<:Integer} =
    vmapreducethen(x -> abs(x)^p, +, x -> exp(one(T)/p * log(abs(x))), A, dims=dims)
_vnorm_impl(A, p::T, dims) where {T<:AbstractFloat} =
    vmapreducethen(x -> exp(p * log(abs(x))), +, x -> exp(one(T)/p * log(abs(x))), A, dims=dims)

"""
    vnorm(A::AbstractArray, p::Real=2; dims=:)

Compute the `p`-norm of `A` along the dimensions `dims` as if the corresponding slices
were vectors.

`p` can assume any numeric value (even though not all values produce a
mathematically valid vector norm). `vnorm(A, Inf)` returns the largest value in `abs.(A)`,
whereas `vnorm(A, -Inf)` returns the smallest; `vnorm(A, 0)` matches the behavior of `norm(A, 0)`.

See also: [`norm`](@ref)
"""
vnorm(A, p::Real=2; dims=:) = p == 0 ? _norm0(A, dims) : _vnorm(A, p, dims)

_vtnorm(A, p::Rational{T}, dims) where {T} = _vtnorm(A, float(p), dims)
Base.@constprop :aggressive function _vtnorm(A, p, dims)
    if p == 2
        vtmapreducethen(abs2, +, √, A, dims=dims)
    elseif p == 1
        vtsum(abs, A, dims=dims)
    elseif p == Inf
        vtmaximum(abs, A, dims=dims)
    elseif p == -Inf
        vtminimum(abs, A, dims=dims)
    else
        _vtnorm_impl(A, p, dims)
    end
end
_vtnorm_impl(A, p::T, dims) where {T<:Integer} =
    vtmapreducethen(x -> abs(x)^p, +, x -> exp(one(T)/p * log(abs(x))), A, dims=dims)
_vtnorm_impl(A, p::T, dims) where {T<:AbstractFloat} =
    vtmapreducethen(x -> exp(p * log(abs(x))), +, x -> exp(one(T)/p * log(abs(x))), A, dims=dims)

"""
    vtnorm(A::AbstractArray, p::Real=2; dims=:)

Compute the `p`-norm of `A` along the dimensions `dims` as if the corresponding slices
were vectors. Threaded.

`p` can assume any numeric value (even though not all values produce a
mathematically valid vector norm). `vnorm(A, Inf)` returns the largest value in `abs.(A)`,
whereas `vnorm(A, -Inf)` returns the smallest; `vnorm(A, 0)` matches the behavior of `norm(A, 0)`.

See also: [`norm`](@ref)
"""
vtnorm(A, p::Real=2; dims=:) = p == 0 ? _norm0(A, dims) : _vtnorm(A, p, dims)
