#
# Date created: 2022-09-14
# Author: aradclif
#
#
############################################################################################

# Statistical things
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

# Assorted functions from StatsBase
function vgeomean(A; dims=:)
    c = 1 / _denom(A, dims)
    vmapreducethen(log, +, x -> exp(c * x), A, dims=dims)
end
function vgeomean(f::F, A; dims=:) where {F}
    c = 1 / _denom(A, dims)
    vmapreducethen(x -> log(f(x)), +, x -> exp(c * x), A, dims=dims)
end

function vharmmean(A; dims=:)
    c = 1 / _denom(A, dims)
    vmapreducethen(inv, +, x -> inv(c * x), A, dims=dims)
end

_xlogx(x::T) where {T} = ifelse(iszero(x), zero(T), x * log(x))
_xlogy(x::T, y::T) where {T} = ifelse(iszero(x) & !isnan(y), zero(T), x * log(y))

ventropy(A; dims=:) = vmapreducethen(_xlogx, +, -, A, dims=dims)
ventropy(A, b::Real; dims=:) = (c = -1 / log(b); vmapreducethen(_xlogx, +, x -> x * c, A, dims=dims ))

vcrossentropy(p, q; dims=:) = vmapreducethen(_xlogy, +, -, p, q, dims=dims)
vcrossentropy(p, q, b::Real; dims=:) = (c = -1 / log(b); vmapreducethen(_xlogy, +, x -> x * c, p, q, dims=dims))

# max-, Shannon, collision and min- entropy assume that p ∈ ℝⁿ, pᵢ ≥ 0, ∑pᵢ=1
_vmaxentropy(p, dims::NTuple{M, Int}) where {M} =
    fill!(similar(p, ntuple(d -> d ∈ dims ? 1 : size(p, d), Val(M))), log(_denom(p, dims)))
_vmaxentropy(p, ::Colon) = log(length(p))
vmaxentropy(p; dims=:) = _vmaxentropy(p, dims)
vshannonentropy(p; dims=:) = vmapreducethen(_xlogx, +, -, p, dims=dims)
vcollisionentropy(p; dims=:) = vmapreducethen(abs2, +, x -> -log(x), p, dims=dims)
vminentropy(p; dims=:) = vmapreducethen(identity, max, x -> -log(x), p, dims=dims)

_vrenyientropy(p, α::T, dims) where {T<:Integer} =
    (c = one(T) / (one(T) - α); vmapreducethen(x -> x^α, +, x -> c * log(x), p, dims=dims))
_vrenyientropy(p, α::T, dims) where {T<:AbstractFloat} =
    (c = one(T) / (one(T) - α); vmapreducethen(x -> exp(α * log(x)), +, x -> c * log(x), p, dims=dims))
_vrenyientropy(p, α::Rational{T}, dims) where {T} = _vrenyientropy(p, float(α), dims)
function vrenyientropy(p, α::Real; dims=:)
    if α ≈ 0
        vmaxentropy(p, dims=dims)
    elseif α ≈ 1
        vshannonentropy(p, dims=dims)
    elseif α ≈ 2
        vcollisionentropy(p, dims=dims)
    elseif isinf(α)
        vminentropy(p, dims=dims)
    else
        _vrenyientropy(p, α, dims)
    end
end
# Loosened restrictions: p ∈ ℝⁿ, pᵢ ≥ 0, ∑pᵢ > 1; that is, if one normalized p, a valid
# probability vector would be produced. Thus, H(x, α) = (α/(1-α)) * (1/α * log∑xᵢ^α - log∑xᵢ)
# H(x, α) = (α / (1 - α)) * ((1/α) * log(sum(z -> z^α, x)) - log(sum(x)))
vrenyientropynorm(p, α::Real; dims=:) =
    vrenyientropy(p, α, dims=dims) .- (α/(1-α)) .* log.(vnorm(p, 1, dims=dims))

vrenyientropy(x2n, 1.5)
renyientropy(x2n, 1.5)

den = sum(abs2, x2)
sum(abs2.(x2)./ den)
sum(abs2, x2 ./ √den)

√(abs2(1 / sum(abs, x2)) * sum(abs2, x2))
(1 / sum(abs, x2)) * norm(x2)
norm(x2n)
norm(x2) / norm(x2, 1)
log(norm(x2))
log(norm(x2)) - log(norm(x2, 1))


# # StatsBase handling of pᵢ = qᵢ = 0
# _xlogxdy(x::T, y::T) where {T} = _xlogy(x, ifelse(iszero(x) & iszero(y), zero(T), x / y))
# vkldivergence(p, q; dims=:) = vvmapreduce(_xlogxdy, +, p, q, dims=dims)
# Slightly more efficient (and likely more stable)
_klterm(x::T, y::T) where {T} = _xlogy(x, x) - _xlogy(x, y)
vkldivergence(p, q; dims=:) = vvmapreduce(_klterm, +, p, q, dims=dims)
vkldivergence(p, q, b::Real; dims=:) = (c = 1 / log(b); vmapreducethen(_klterm, +, x -> x * c, p, q, dims=dims))




vcounteq(x, y; dims=:) = vvmapreduce(==, +, x, y, dims=dims)
vtcounteq(x, y; dims=:) = vtmapreduce(==, +, x, y, dims=dims)
vcountne(x, y; dims=:) = vvmapreduce(!=, +, x, y, dims=dims)
vtcountne(x, y; dims=:) = vtmapreduce(!=, +, x, y, dims=dims)

function vmeanad(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vmapreducethen((xᵢ, yᵢ) -> abs(xᵢ - yᵢ) , +, z -> c * z, x, y, dims=dims)
end
function vtmeanad(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vtmapreducethen((xᵢ, yᵢ) -> abs(xᵢ - yᵢ) , +, z -> c * z, x, y, dims=dims)
end

vmaxad(x, y; dims=:) = vvmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ) , max, x, y, dims=dims)
vtmaxad(x, y; dims=:) = vtmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ) , max, x, y, dims=dims)


# generalized KL divergence sum(a*log(a/b)-a+b)
vgkldiv(x, y; dims=:) = vvmapreduce((xᵢ, yᵢ) -> xᵢ * (log(xᵢ) - log(yᵢ)) - xᵢ + yᵢ, +, x, y, dims=dims)
vtgkldiv(x, y; dims=:) = vtmapreduce((xᵢ, yᵢ) -> xᵢ * (log(xᵢ) - log(yᵢ)) - xᵢ + yᵢ, +, x, y, dims=dims)
