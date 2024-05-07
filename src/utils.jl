#
# Date created: 2022-09-14
# Author: aradclif
#
#
############################################################################################

#### Array size/shape functions
_denom(A, dims) = prod(d -> size(A, d), dims)
_denom(A, ::Colon) = length(A)

@inline _reducedsize(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M} = ntuple(d -> d ∈ dims ? 1 : size(A, d), Val(N))

#### Numerical stability/limits
function _xlogx(x::T) where {T<:AbstractFloat}
    ifelse(iszero(x), zero(T), x * log(x))
end
_xlogx(x::Real) = _xlogx(float(x))

function _xlogy(x::T, y::T) where {T<:AbstractFloat}
    ifelse(iszero(x) & !isnan(y), zero(T), x * log(y))
end
_xlogy(x::Real, y::Real) = _xlogy(float(x), float(y))
# _xlogxdy(x::T, y::T) where {T} = _xlogy(x, ifelse(iszero(x) & iszero(y), zero(T), x / y))
function _klterm(x::T, y::T) where {T}
    xf = float(x)
    yf = float(y)
    _xlogy(xf, xf) - _xlogy(xf, yf)
end
