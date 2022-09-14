#
# Date created: 2022-09-13
# Author: aradclif
#
#
############################################################################################
_denom(A, dims) = prod(d -> size(A, d), dims)
_denom(A, ::Colon) = length(A)

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
    vmapreducethen(x -> abs(x)^p, +, x -> x^(one(T)/p), A, dims=dims)
_vnorm_impl(A, p::T, dims) where {T<:AbstractFloat} =
    vmapreducethen(x -> exp(p * log(abs(x))), +, x -> exp(one(T)/p * log(abs(x))), A, dims=dims)

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
    vtmapreducethen(x -> abs(x)^p, +, x -> x^(one(T)/p), A, dims=dims)
_vtnorm_impl(A, p::T, dims) where {T<:AbstractFloat} =
    vtmapreducethen(x -> exp(p * log(abs(x))), +, x -> exp(one(T)/p * log(abs(x))), A, dims=dims)

vtnorm(A, p::Real=2; dims=:) = p == 0 ? _norm0(A, dims) : _vtnorm(A, p, dims)

# Distances
veuclidean(x, y; dims=:) = vmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, √, x, y, dims=dims)
vteuclidean(x, y; dims=:) = vtmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, √, x, y, dims=dims)
vmanhattan(x, y; dims=:) = vvmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), +, x, y, dims=dims)
vtmanhattan(x, y; dims=:) = vtmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), +, x, y, dims=dims)

function vmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> c * z, x, y, dims=dims)
end
function vtmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vtmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> c * z, x, y, dims=dims)
end
function vrmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> √(c * z), x, y, dims=dims)
end
function vtrmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vtmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> √(c * z), x, y, dims=dims)
end

function vmean(f, A; dims=:)
    c = 1 / _denom(A, dims)
    vmapreducethen(f, +, x -> c * x, A, dims=dims)
end
vmean(A; dims=:) = vmean(identity, A, dims=dims)

function vtmean(f, A; dims=:)
    c = 1 / _denom(A, dims)
    vtmapreducethen(f, +, x -> c * x, A, dims=dims)
end
vtmean(A; dims=:) = vtmean(identity, A, dims=dims)

# Naturally, faster than the overflow/underflow-safe logsumexp, but if one can tolerate it...
vlse(A; dims=:) = vmapreducethen(exp, +, log, A, dims=dims)
vtlse(A; dims=:) = vtmapreducethen(exp, +, log, A, dims=dims)
