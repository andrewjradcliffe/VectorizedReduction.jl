#
# Date created: 2022-09-14
# Author: aradclif
#
#
############################################################################################
# Assorted functions from StatsBase, plus some of my own

################
# Means
function vmean(f::F, A; dims=:) where {F}
    c = 1 / _denom(A, dims)
    vmapreducethen(f, +, x -> c * x, A, dims=dims)
end
vmean(A; dims=:) = vmean(identity, A, dims=dims)

function vtmean(f::F, A; dims=:) where {F}
    c = 1 / _denom(A, dims)
    vtmapreducethen(f, +, x -> c * x, A, dims=dims)
end
vtmean(A; dims=:) = vtmean(identity, A, dims=dims)

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

function vtgeomean(A; dims=:)
    c = 1 / _denom(A, dims)
    vtmapreducethen(log, +, x -> exp(c * x), A, dims=dims)
end
function vtgeomean(f::F, A; dims=:) where {F}
    c = 1 / _denom(A, dims)
    vtmapreducethen(x -> log(f(x)), +, x -> exp(c * x), A, dims=dims)
end
function vtharmmean(A; dims=:)
    c = 1 / _denom(A, dims)
    vtmapreducethen(inv, +, x -> inv(c * x), A, dims=dims)
end

# Mean on the log scale
function vmean_log(f::F, A; dims=:) where {F}
    c = log(_denom(A, dims))
    vmapreducethen(f, +, x -> log(x) - c, A, dims=dims)
end
function vtmean_log(f::F, A; dims=:) where {F}
    c = log(_denom(A, dims))
    vtmapreducethen(f, +, x -> log(x) - c, A, dims=dims)
end

################
# logsumexp (the naive and unsafe version)
# Naturally, faster than the overflow/underflow-safe logsumexp, but if one can tolerate it...
vlse(A; dims=:) = vmapreducethen(exp, +, log, A, dims=dims)
vtlse(A; dims=:) = vtmapreducethen(exp, +, log, A, dims=dims)
vlse(f::F, A; dims=:) where {F} = vmapreducethen(x -> exp(f(x)), +, log, A, dims=dims)
vtlse(f::F, A; dims=:) where {F} = vtmapreducethen(x -> exp(f(x)), +, log, A, dims=dims)

function vlse_mean(A; dims=:)
    c = log(_denom(A, dims))
    vmapreducethen(exp, +, x -> log(x) - c, A, dims=dims)
end
function vlse_mean(f::F, A; dims=:) where {F}
    c = log(_denom(A, dims))
    vmapreducethen(x -> exp(f(x)), +, x -> log(x) - c, A, dims=dims)
end

function vtlse_mean(A; dims=:)
    c = log(_denom(A, dims))
    vtmapreducethen(exp, +, x -> log(x) - c, A, dims=dims)
end
function vtlse_mean(f::F, A; dims=:) where {F}
    c = log(_denom(A, dims))
    vtmapreducethen(x -> exp(f(x)), +, x -> log(x) - c, A, dims=dims)
end

################
# Entropies

ventropy(A; dims=:) = vshannonentropy(A, dims=dims)
ventropy(A, b::Real; dims=:) = (c = -1 / log(b); vmapreducethen(_xlogx, +, x -> x * c, A, dims=dims ))

vcrossentropy(p, q; dims=:) = vmapreducethen(_xlogy, +, -, p, q, dims=dims)
vcrossentropy(p, q, b::Real; dims=:) = (c = -1 / log(b); vmapreducethen(_xlogy, +, x -> x * c, p, q, dims=dims))

# max-, Shannon, collision and min- entropy assume that p ∈ ℝⁿ, pᵢ ≥ 0, ∑pᵢ=1
_vmaxentropy(p, dims::NTuple{M, Int}) where {M} =
    fill!(similar(p, _reducedsize(p, dims)), log(_denom(p, dims)))
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
    α < 0 && throw(ArgumentError("Order of Rényi entropy not legal, $(α) < 0."))
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
# vrenyientropynorm(p, α::Real; dims=:) =
#     vrenyientropy(p, α, dims=dims) .- (α/(1-α)) .* log.(vnorm(p, 1, dims=dims))

####

vtentropy(A; dims=:) = vtshannonentropy(A, dims=dims)
vtentropy(A, b::Real; dims=:) = (c = -1 / log(b); vtmapreducethen(_xlogx, +, x -> x * c, A, dims=dims ))

vtcrossentropy(p, q; dims=:) = vtmapreducethen(_xlogy, +, -, p, q, dims=dims)
vtcrossentropy(p, q, b::Real; dims=:) = (c = -1 / log(b); vtmapreducethen(_xlogy, +, x -> x * c, p, q, dims=dims))

# threaded versions
_vtmaxentropy(p, dims::NTuple{M, Int}) where {M} =
    fill!(similar(p, _reducedsize(p, dims)), log(_denom(p, dims)))
_vtmaxentropy(p, ::Colon) = log(length(p))
vtmaxentropy(p; dims=:) = _vtmaxentropy(p, dims)
vtshannonentropy(p; dims=:) = vtmapreducethen(_xlogx, +, -, p, dims=dims)
vtcollisionentropy(p; dims=:) = vtmapreducethen(abs2, +, x -> -log(x), p, dims=dims)
vtminentropy(p; dims=:) = vtmapreducethen(identity, max, x -> -log(x), p, dims=dims)

_vtrenyientropy(p, α::T, dims) where {T<:Integer} =
    (c = one(T) / (one(T) - α); vtmapreducethen(x -> x^α, +, x -> c * log(x), p, dims=dims))
_vtrenyientropy(p, α::T, dims) where {T<:AbstractFloat} =
    (c = one(T) / (one(T) - α); vtmapreducethen(x -> exp(α * log(x)), +, x -> c * log(x), p, dims=dims))
_vtrenyientropy(p, α::Rational{T}, dims) where {T} = _vtrenyientropy(p, float(α), dims)
function vtrenyientropy(p, α::Real; dims=:)
    α < 0 && throw(ArgumentError("Order of Rényi entropy not legal, $(α) < 0."))
    if α ≈ 0
        vtmaxentropy(p, dims=dims)
    elseif α ≈ 1
        vtshannonentropy(p, dims=dims)
    elseif α ≈ 2
        vtcollisionentropy(p, dims=dims)
    elseif isinf(α)
        vtminentropy(p, dims=dims)
    else
        _vtrenyientropy(p, α, dims)
    end
end

################
# Divergences

# Kullback-Leibler
# # StatsBase handling of pᵢ = qᵢ = 0
# vkldivergence(p, q; dims=:) = vvmapreduce(_xlogxdy, +, p, q, dims=dims)
# Slightly more efficient (and likely more stable)
vkldivergence(p, q; dims=:) = vvmapreduce(_klterm, +, p, q, dims=dims)
vkldivergence(p, q, b::Real; dims=:) = (c = 1 / log(b); vmapreducethen(_klterm, +, x -> x * c, p, q, dims=dims))
vtkldivergence(p, q; dims=:) = vtmapreduce(_klterm, +, p, q, dims=dims)
vtkldivergence(p, q, b::Real; dims=:) = (c = 1 / log(b); vtmapreducethen(_klterm, +, x -> x * c, p, q, dims=dims))

# generalized KL divergence sum(a*log(a/b)-a+b)
vgkldiv(x, y; dims=:) = vvmapreduce((xᵢ, yᵢ) -> xᵢ * (log(xᵢ) - log(yᵢ)) - xᵢ + yᵢ, +, x, y, dims=dims)
vtgkldiv(x, y; dims=:) = vtmapreduce((xᵢ, yᵢ) -> xᵢ * (log(xᵢ) - log(yᵢ)) - xᵢ + yᵢ, +, x, y, dims=dims)

# Rénya
_vrenyidivergence(p, q, α::Real, dims) =
    vmapreducethen((pᵢ, qᵢ) -> pᵢ^α / qᵢ^(α-1), +, x -> (1/(α-1)) * log(x), p, q, dims=dims)
_vrenyidivergence(p, q, α::Rational{T}, dims) where {T} = _vrenyidivergence(p, q, float(α), dims)

function vrenyidivergence(p, q, α::Real; dims=:)
    if α ≈ 0
        vmapreducethen((pᵢ, qᵢ) -> ifelse(pᵢ > zero(pᵢ), qᵢ, zero(qᵢ)), +, x -> -log(x), p, q, dims=dims)
    elseif α ≈ 0.5
        vmapreducethen((pᵢ, qᵢ) -> √(pᵢ * qᵢ), +, x -> -2log(x), p, q, dims=dims)
    elseif α ≈ 1
        vkldivergence(p, q, dims=dims)
    elseif α ≈ 2
        c = log(_denom(p, dims))
        vmapreducethen(/, +, x -> log(x) - c, p, q, dims=dims)
    elseif isinf(α)
        vmapreducethen(/, max, log, p, q, dims=dims)
    else
        _vrenyidivergence(p, q, α, dims)
    end
end

_vtrenyidivergence(p, q, α::Real, dims) =
    vtmapreducethen((pᵢ, qᵢ) -> pᵢ^α / qᵢ^(α-1), +, x -> (1/(α-1)) * log(x), p, q, dims=dims)
_vtrenyidivergence(p, q, α::Rational{T}, dims) where {T} = _vtrenyidivergence(p, q, float(α), dims)

function vtrenyidivergence(p, q, α::Real; dims=:)
    if α ≈ 0
        vtmapreducethen((pᵢ, qᵢ) -> ifelse(pᵢ > zero(pᵢ), qᵢ, zero(qᵢ)), +, x -> -log(x), p, q, dims=dims)
    elseif α ≈ 0.5
        vtmapreducethen((pᵢ, qᵢ) -> √(pᵢ * qᵢ), +, x -> -2log(x), p, q, dims=dims)
    elseif α ≈ 1
        vkldivergence(p, q, dims=dims)
    elseif α ≈ 2
        c = log(_denom(p, dims))
        vtmapreducethen(/, +, x -> log(x) - c, p, q, dims=dims)
    elseif isinf(α)
        vtmapreducethen(/, max, log, p, q, dims=dims)
    else
        _vtrenyidivergence(p, q, α, dims)
    end
end

################
# Deviations

"""
    vcounteq(x::AbstractArray, y::AbstractArray; dims=:)

Count the number of elements for which `xᵢ == yᵢ` returns `true` on the vectors
corresponding to the slices along the dimension `dims`.

See also: [`vcountne`](@ref)
"""
vcounteq(x, y; dims=:) = vvmapreduce(==, +, x, y, dims=dims)

"""
    vtcounteq(x::AbstractArray, y::AbstractArray; dims=:)

Count the number of elements for which `xᵢ == yᵢ` returns `true` on the vectors
corresponding to the slices along the dimension `dims`. Threaded.

See also: [`vtcountne`](@ref)
"""
vtcounteq(x, y; dims=:) = vtmapreduce(==, +, x, y, dims=dims)

"""
    vcountne(x::AbstractArray, y::AbstractArray; dims=:)

Count the number of elements for which `xᵢ != yᵢ` returns `true` on the vectors
corresponding to the slices along the dimension `dims`.

See also: [`vcounteq`](@ref)
"""
vcountne(x, y; dims=:) = vvmapreduce(!=, +, x, y, dims=dims)

"""
    vtcountne(x::AbstractArray, y::AbstractArray; dims=:)

Count the number of elements for which `xᵢ != yᵢ` returns `true` on the vectors
corresponding to the slices along the dimension `dims`. Threaded.

See also: [`vtcounteq`](@ref)
"""
vtcountne(x, y; dims=:) = vtmapreduce(!=, +, x, y, dims=dims)

"""
    vmeanad(x::AbstractArray, y::AbstractArray; dims=:)

Compute the mean absolute deviation between the vectors corresponding to the slices along
the dimension `dims`.

See also: [`vmaxad`](@ref)
"""
function vmeanad(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vmapreducethen((xᵢ, yᵢ) -> abs(xᵢ - yᵢ) , +, z -> c * z, x, y, dims=dims)
end

"""
    vtmeanad(x::AbstractArray, y::AbstractArray; dims=:)

Compute the mean absolute deviation between the vectors corresponding to the slices along
the dimension `dims`. Threaded.

See also: [`vtmaxad`](@ref)
"""
function vtmeanad(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vtmapreducethen((xᵢ, yᵢ) -> abs(xᵢ - yᵢ) , +, z -> c * z, x, y, dims=dims)
end

"""
    vmaxad(x::AbstractArray, y::AbstractArray; dims=:)

Compute the maximum absolute deviation between the vectors corresponding to the slices along
the dimension `dims`.

See also: [`vmeanad`](@ref)
"""
vmaxad(x, y; dims=:) = vvmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ) , max, x, y, dims=dims)

"""
    vtmaxad(x::AbstractArray, y::AbstractArray; dims=:)

Compute the maximum absolute deviation between the vectors corresponding to the slices along
the dimension `dims`. Threaded.

See also: [`vtmeanad`](@ref)
"""
vtmaxad(x, y; dims=:) = vtmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ) , max, x, y, dims=dims)

"""
    vmse(x::AbstractArray, y::AbstractArray; dims=:)

Compute the mean squared error between the vectors corresponding to the slices along
the dimension `dims`.

See also: [`vrmse`](@ref)
"""
function vmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> c * z, x, y, dims=dims)
end

"""
    vtmse(x::AbstractArray, y::AbstractArray; dims=:)

Compute the mean squared error between the vectors corresponding to the slices along
the dimension `dims`. Threaded.

See also: [`vtrmse`](@ref)
"""
function vtmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vtmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> c * z, x, y, dims=dims)
end

"""
    vrmse(x::AbstractArray, y::AbstractArray; dims=:)

Compute the square root of the mean squared error between the vectors corresponding
to the slices along the dimension `dims`. More efficient than `sqrt.(vmse(...))`
as the `sqrt` operation is performed at the point of generation, thereby eliminating the
full traversal which would otherwise occur.

See also: [`vmse`](@ref)
"""
function vrmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> √(c * z), x, y, dims=dims)
end

"""
    vtrmse(x::AbstractArray, y::AbstractArray; dims=:)

Compute the square root of the mean squared error between the vectors corresponding
to the slices along the dimension `dims`. More efficient than `sqrt.(vmse(...))`
as the `sqrt` operation is performed at the point of generation, thereby eliminating the
full traversal which would otherwise occur. Threaded.

See also: [`vtmse`](@ref)
"""
function vtrmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vtmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> √(c * z), x, y, dims=dims)
end

# Match names with StatsBase
const vmsd = vmse
const vtmsd = vtmse
const vrmsd = vrmse
const vtrmsd = vtrmse

################
