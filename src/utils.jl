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
_xlogx(x::T) where {T} = ifelse(iszero(x), zero(T), x * log(x))
_xlogy(x::T, y::T) where {T} = ifelse(iszero(x) & !isnan(y), zero(T), x * log(y))
# _xlogxdy(x::T, y::T) where {T} = _xlogy(x, ifelse(iszero(x) & iszero(y), zero(T), x / y))
_klterm(x::T, y::T) where {T} = _xlogy(x, x) - _xlogy(x, y)
