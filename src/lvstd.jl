#
# Date created: 2022-03-18
# Author: aradclif
#
#
############################################################################################
################
# function smul_sqrt_inplacebody(N::Int)
#     body = Expr(:block)
#     a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
#     e = Expr(:(=), a, Expr(:call, :sqrt, Expr(:call, :*, a, :x)))
#     push!(body.args, e)
#     body
# end

# function smul_sqrt_inplace_quote(N::Int)
#     ls = loopgen(N)
#     body = smul_sqrt_inplacebody(N)
#     push!(ls.args, body)
#     return quote
#         @turbo $ls
#         return A
#     end
# end

# @generated function smul_sqrt!(A::AbstractArray{T, N}, x::T) where {T, N}
#     smul_sqrt_inplace_quote(N)
# end

function _smul_sqrt!(A::AbstractArray{T, N}, x::T) where {T, N}
    @turbo for i ∈ eachindex(A)
        A[i] = sqrt(A[i] * x)
    end
    A
end

function _lvstd(A::AbstractArray{T, N}, dims::NTuple{M, Int}, corrected::Bool) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        C = hvncat(ntuple(_ -> 1, Val(N)), true, lvstd1(A, corrected))
    else
        B = lvmean(A, dims=dims, multithreaded=false)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(B, d), Val(N))
        C = zeros(Base.promote_op(/, T, Int), Dᴮ′)
        sumsqdiff!(C, A, B, Dᴮ′)
        denom = 1
        for d = 1:N
            denom = d ∈ dims ? denom * size(A, d) : denom
        end
        denom = corrected ? denom - 1 : denom
        x = inv(denom)
        _smul_sqrt!(C, x)
    end
    return C
end
_lvstd(A::AbstractArray{T, N}, dims::Int, corrected) where {T, N} = _lvstd(A, (dims,), corrected)
_lvstd(A::AbstractArray{T, N}, ::Colon, corrected) where {T, N} = lvstd1(A, corrected)
lvstd1(A::AbstractArray{T, N}, corrected::Bool=true) where {T, N} = √(lvvar1(A, corrected))

################ threaded version
function _tsmul_sqrt!(A::AbstractArray{T, N}, x::T) where {T, N}
    @tturbo for i ∈ eachindex(A)
        A[i] = sqrt(A[i] * x)
    end
    A
end

function _lvtstd(A::AbstractArray{T, N}, dims::NTuple{M, Int}, corrected::Bool) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        C = hvncat(ntuple(_ -> 1, Val(N)), true, lvtstd1(A, corrected))
    else
        B = lvmean(A, dims=dims, multithreaded=true)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(B, d), Val(N))
        C = zeros(Base.promote_op(/, T, Int), Dᴮ′)
        tsumsqdiff!(C, A, B, Dᴮ′)
        denom = 1
        for d = 1:N
            denom = d ∈ dims ? denom * size(A, d) : denom
        end
        denom = corrected ? denom - 1 : denom
        x = inv(denom)
        _tsmul_sqrt!(C, x)
    end
    return C
end
_lvtstd(A::AbstractArray{T, N}, dims::Int, corrected) where {T, N} = _lvtstd(A, (dims,), corrected)
_lvtstd(A::AbstractArray{T, N}, ::Colon, corrected) where {T, N} = lvtstd1(A, corrected)
lvtstd1(A::AbstractArray{T, N}, corrected::Bool=true) where {T, N} = √(lvtvar1(A, corrected))

################
# Common interface
function lvstd(A::AbstractArray{T, N}; dims=:, corrected::Bool=true, multithreaded=:auto) where {T, N}
    if (multithreaded === :auto && length(A) > 4095) || multithreaded === true
        μ = _lvtstd(A, dims, corrected)
    else
        μ = _lvstd(A, dims, corrected)
    end
    return μ
end
