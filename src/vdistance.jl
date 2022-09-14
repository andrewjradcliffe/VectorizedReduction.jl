#
# Date created: 2022-09-14
# Author: aradclif
#
#
############################################################################################

# Distances
_vminkowski(x, y, p::T, dims=:) where {T<:Integer} =
    vmapreducethen((xᵢ, yᵢ) -> abs(xᵢ - yᵢ)^p, +, x -> exp(one(T)/p * log(abs(x))), x, y, dims=dims)
_vminkowski(x, y, p::T, dims=:) where {T<:AbstractFloat} =
    vmapreducethen((xᵢ, yᵢ) -> exp(p * log(abs(xᵢ - yᵢ))), +, x -> exp(one(T)/p * log(abs(x))), x, y, dims=dims)
_vminkowski(x, y, p::Rational{T}, dims=:) where {T} = _vminkowski(x, y, float(p), dims=dims)
function vminkowski(x, y, p; dims=:)
    if p == 2
        veuclidean(x, y, dims=dims)
    elseif p == 1
        vmanhattan(x, y, dims=dims)
    elseif p == Inf
        vchebyshev(x, y, dims=dims)
    elseif p == -Inf
        vchebyshev₋(x, y, dims=dims)
    else
        _vminkowski(x, y, p, dims)
    end
end

vmanhattan(x, y; dims=:) = vvmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), +, x, y, dims=dims)
vtmanhattan(x, y; dims=:) = vtmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), +, x, y, dims=dims)
veuclidean(x, y; dims=:) = vmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, √, x, y, dims=dims)
vteuclidean(x, y; dims=:) = vtmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, √, x, y, dims=dims)
vchebyshev(x, y; dims=:) = vvmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), max, x, y, dims=dims)
vchebyshev₋(x, y; dims=:) = vvmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), min, x, y, dims=dims)
vtchebyshev(x, y; dims=:) = vtmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), max, x, y, dims=dims)
vtchebyshev₋(x, y; dims=:) = vtmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), min, x, y, dims=dims)
