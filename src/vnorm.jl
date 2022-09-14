#
# Date created: 2022-09-13
# Author: aradclif
#
#
############################################################################################
_denom(A, dims) = prod(d -> size(A, d), dims)
_denom(A, ::Colon) = length(A)


veuclidean(x, y; dims=:) = vmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, √, x, y, dims=dims)
vrmse(x, y; dims=:) = vmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, √, x, y, dims=dims)

_norm0(A, dims::NTuple{M, Int}) where {M} =
    fill!(similar(A, ntuple(d -> d ∈ dims ? 1 : size(A, d), Val(M))), _denom(A, dims))
_norm0(A, ::Colon) = float(length(A))

# Base.@constprop :aggressive function _vnorm(A, p::T, dims) where {T<:AbstractFloat}
#     if p === one(T)
#         vvsum(abs, A, dims=dims)
#     elseif p === 2one(T)
#         vmapreducethen(abs2, +, √, A, dims=dims)
#     elseif p === typemax(T)
#         vvmaximum(abs, A, dims=dims)
#     elseif p === typemin(T)
#         vvminimum(abs, A, dims=dims)
#     else
#         vmapreducethen(x -> exp(p * log(abs(x))), +, x -> exp(1/p * log(abs(x))), A, dims=dims)
#     end
# end
# function _vnorm(A, p::Integer, dims)
#     if p == 1
#         vvsum(abs, A, dims=dims)
#     elseif p == 2
#         vmapreducethen(abs2, +, √, A, dims=dims)
#     else
#         vmapreducethen(x -> abs(x)^p, +, x -> x^(1/p), A, dims=dims)
#     end
# end
# vnorm(A, p::Real=2; dims=:) = p == 0 ? _norm0(A, dims) : _vnorm(A, p, dims)

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


_vmean(f, g, A, dims) = vmapreducethen(f, +, g, A, dims=dims)
vmean(f, A; dims=:) = (c = 1 / _denom(A, dims); _vmean(f, x -> c * x, A, dims))
vmean(A; dims=:) = vmean(identity, A, dims=dims)



euclidean(x, y; dims=:) = .√mapreduce((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, x, y, dims=dims)
vlse(A; dims=:) = vmapreducethen(exp, +, log, A, dims=dims)
