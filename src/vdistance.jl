#
# Date created: 2022-09-14
# Author: aradclif
#
#
############################################################################################

# Distances
_vminkowski(x, y, p::T, dims=:) where {T<:Integer} =
    vmapreducethen((xᵢ, yᵢ) -> abs(xᵢ - yᵢ)^p, +, x -> exp((one(T)/p) * log(abs(x))), x, y, dims=dims)
_vminkowski(x, y, p::T, dims=:) where {T<:AbstractFloat} =
    vmapreducethen((xᵢ, yᵢ) -> exp(p * log(abs(xᵢ - yᵢ))), +, x -> exp((one(T)/p) * log(abs(x))), x, y, dims=dims)
_vminkowski(x, y, p::Rational{T}, dims=:) where {T} = _vminkowski(x, y, float(p), dims=dims)

"""
    vminkowski(x::AbstractArray, y::AbstractArray, p::Real; dims=:)

Compute the Minkowski distance between the vectors corresponding to the slices along the
dimensions `dims`.

`p` can assume any numeric value (even though not all values produce a
mathematically valid vector norm). `vminkowski(x, y, Inf)` returns the largest value in
`abs.(x .- y)`, whereas `vminkowski(x, y, -Inf)` returns the smallest.

See also: [`vmanhattan`](@ref), [`veuclidean`](@ref), [`vchebyshev`](@ref)
"""
function vminkowski(x, y, p::Real; dims=:)
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

_vtminkowski(x, y, p::T, dims=:) where {T<:Integer} =
    vtmapreducethen((xᵢ, yᵢ) -> abs(xᵢ - yᵢ)^p, +, x -> exp((one(T)/p) * log(abs(x))), x, y, dims=dims)
_vtminkowski(x, y, p::T, dims=:) where {T<:AbstractFloat} =
    vtmapreducethen((xᵢ, yᵢ) -> exp(p * log(abs(xᵢ - yᵢ))), +, x -> exp((one(T)/p) * log(abs(x))), x, y, dims=dims)
_vtminkowski(x, y, p::Rational{T}, dims=:) where {T} = _vtminkowski(x, y, float(p), dims=dims)

"""
    vtminkowski(x::AbstractArray, y::AbstractArray, p::Real; dims=:)

Compute the Minkowski distance between the vectors corresponding to the slices along the
dimensions `dims`. Threaded.

`p` can assume any numeric value (even though not all values produce a
mathematically valid vector norm). `vtminkowski(x, y, Inf)` returns the largest value in
`abs.(x .- y)`, whereas `vtminkowski(x, y, -Inf)` returns the smallest.

See also: [`vtmanhattan`](@ref), [`vteuclidean`](@ref), [`vtchebyshev`](@ref)
"""
function vtminkowski(x, y, p::Real; dims=:)
    if p == 2
        vteuclidean(x, y, dims=dims)
    elseif p == 1
        vtmanhattan(x, y, dims=dims)
    elseif p == Inf
        vtchebyshev(x, y, dims=dims)
    elseif p == -Inf
        vtchebyshev₋(x, y, dims=dims)
    else
        _vtminkowski(x, y, p, dims)
    end
end

"""
    vmanhattan(x::AbstractArray, y::AbstractArray; dims=:)

Compute the Manhattan distance between the vectors corresponding to the slices along the
dimensions `dims`.

See also: [`veuclidean`](@ref), [`vchebyshev`](@ref), [`vminkowski`](@ref)
"""
vmanhattan(x, y; dims=:) = vvmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), +, x, y, dims=dims)

"""
    vtmanhattan(x::AbstractArray, y::AbstractArray; dims=:)

Compute the Manhattan distance between the vectors corresponding to the slices along the
dimensions `dims`. Threaded.

See also: [`vteuclidean`](@ref), [`vtchebyshev`](@ref), [`vtminkowski`](@ref)
"""
vtmanhattan(x, y; dims=:) = vtmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), +, x, y, dims=dims)

"""
    veuclidean(x::AbstractArray, y::AbstractArray; dims=:)

Compute the Euclidean distance between the vectors corresponding to the slices along the
dimensions `dims`.

See also: [`vmanhattan`](@ref), [`vchebyshev`](@ref), [`vminkowski`](@ref)
"""
veuclidean(x, y; dims=:) = vmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, √, x, y, dims=dims)

"""
    vteuclidean(x::AbstractArray, y::AbstractArray; dims=:)

Compute the Euclidean distance between the vectors corresponding to the slices along the
dimensions `dims`. Threaded.

See also: [`vtmanhattan`](@ref), [`vtchebyshev`](@ref), [`vtminkowski`](@ref)
"""
vteuclidean(x, y; dims=:) = vtmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, √, x, y, dims=dims)

"""
    vchebyshev(x::AbstractArray, y::AbstractArray; dims=:)

Compute the Chebyshev distance between the vectors corresponding to the slices along the
dimensions `dims`.

See also: [`vmanhattan`](@ref), [`veuclidean`](@ref), [`vminkowski`](@ref)
"""
vchebyshev(x, y; dims=:) = vvmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), max, x, y, dims=dims)
vchebyshev₋(x, y; dims=:) = vvmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), min, x, y, dims=dims)

"""
    vtchebyshev(x::AbstractArray, y::AbstractArray; dims=:)

Compute the Chebyshev distance between the vectors corresponding to the slices along the
dimensions `dims`. Threaded.

See also: [`vtmanhattan`](@ref), [`vteuclidean`](@ref), [`vtminkowski`](@ref)
"""
vtchebyshev(x, y; dims=:) = vtmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), max, x, y, dims=dims)
vtchebyshev₋(x, y; dims=:) = vtmapreduce((xᵢ, yᵢ) -> abs(xᵢ - yᵢ), min, x, y, dims=dims)
