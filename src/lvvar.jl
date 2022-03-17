#
# Date created: 2022-03-09
# Author: aradclif
#
#
############################################################################################

function sumsqdiff_body(N::Int, D)
    body = Expr(:block)
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    params = D.parameters
    b = Expr(:ref, :B, ntuple(d -> params[d] == Static.One ? 1 : Symbol(:i_, d), N)...)
    c = Expr(:ref, :C, ntuple(d -> params[d] == Static.One ? 1 : Symbol(:i_, d), N)...)
    δ = Expr(:(=), :Δ, Expr(:call, :-, a, b))
    e = Expr(:(=), c, Expr(:call, :+, c, Expr(:call, :*, :Δ, :Δ)))
    push!(body.args, δ)
    push!(body.args, e)
    body
end

function sumsqdiff_quote(N::Int, D)
    ls = loopgen(N)
    body = sumsqdiff_body(N, D)
    push!(ls.args, body)
    return quote
        @turbo $ls
        return C
    end
end
@generated function sumsqdiff!(C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N},
                               B::AbstractArray{Tₘ, N}, dims::D) where {Tₒ, T, Tₘ, N, D}
    sumsqdiff_quote(N, D)
end

function _lvvar(A::AbstractArray{T, N}, dims::NTuple{M, Int}, corrected::Bool) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        C = hvncat(ntuple(_ -> 1, Val(N)), true, lvvar1(A, corrected))
    else
        B = lvmean(A, dims=dims, multithreaded=false)
        Dᴮ = size(B)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : Dᴮ[d], Val(N))
        C = zeros(Base.promote_op(/, T, Int), Dᴮ)
        sumsqdiff!(C, A, B, Dᴮ′)
        Dᴬ = size(A)
        denom = 1
        for d ∈ eachindex(Dᴬ)
            denom = d ∈ dims ? denom * Dᴬ[d] : denom
        end
        denom = corrected ? denom - 1 : denom
        x = inv(denom)
        _smul!(C, x)
    end
    return C
end
_lvvar(A::AbstractArray{T, N}, dims::Int, corrected::Bool) where {T, N} =
    _lvvar(A, (dims,), corrected)
_lvvar(A::AbstractArray{T, N}, ::Colon, corrected::Bool) where {T, N} = lvvar1(A, corrected)

function lvvar1(A::AbstractArray{T, N}, corrected::Bool=true) where {T, N}
    s = zero(Base.promote_op(+, T, Int))
    @turbo for i ∈ eachindex(A)
        s += A[i]
    end
    μ = s / length(A)
    ss = zero(Base.promote_op(/, T, Int))
    @turbo for i ∈ eachindex(A)
        Δ = A[i] - μ
        ss += Δ * Δ
    end
    return corrected ? ss / (length(A) - 1) : ss / length(A)
end

################ threaded version

function tsumsqdiff_quote(N::Int, D)
    ls = loopgen(N)
    body = sumsqdiff_body(N, D)
    push!(ls.args, body)
    return quote
        @tturbo $ls
        return C
    end
end
@generated function tsumsqdiff!(C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N},
                                B::AbstractArray{Tₘ, N}, dims::D) where {Tₒ, T, Tₘ, N, D}
    tsumsqdiff_quote(N, D)
end

function _lvtvar(A::AbstractArray{T, N}, dims::NTuple{M, Int}, corrected::Bool) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        C = hvncat(ntuple(_ -> 1, Val(N)), true, lvtvar1(A, corrected))
    else
        B = lvmean(A, dims=dims, multithreaded=true)
        Dᴮ = size(B)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : Dᴮ[d], N)
        C = zeros(Base.promote_op(/, T, Int), Dᴮ)
        tsumsqdiff!(C, A, B, Dᴮ′)
        Dᴬ = size(A)
        denom = 1
        for d ∈ eachindex(Dᴬ)
            denom = d ∈ dims ? denom * Dᴬ[d] : denom
        end
        denom = corrected ? denom - 1 : denom
        x = inv(denom)
        _tsmul!(C, x)
    end
    return C
end
_lvtvar(A::AbstractArray{T, N}, dims::Int, corrected::Bool) where {T, N} =
    _lvtvar(A, (dims,), corrected)
_lvtvar(A::AbstractArray{T, N}, ::Colon, corrected::Bool) where {T, N} = lvtvar1(A, corrected)

function lvtvar1(A::AbstractArray{T, N}, corrected::Bool=true) where {T, N}
    s = zero(Base.promote_op(+, T, Int))
    @tturbo for i ∈ eachindex(A)
        s += A[i]
    end
    μ = s / length(A)
    ss = zero(Base.promote_op(/, T, Int))
    @tturbo for i ∈ eachindex(A)
        Δ = A[i] - μ
        ss += Δ * Δ
    end
    return corrected ? ss / (length(A) - 1) : ss / length(A)
end

################
# Common interface
function lvvar(A::AbstractArray{T, N}; dims=:, corrected::Bool=true, multithreaded=:auto) where {T, N}
    if (multithreaded === :auto && length(A) > 4095) || multithreaded === true
        μ = _lvtvar(A, dims, corrected)
    else
        μ = _lvvar(A, dims, corrected)
    end
    return μ
end

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

function smul_sqrt!(A::AbstractArray{T, N}, x::T) where {T, N}
    @turbo for i ∈ eachindex(A)
        A[i] = sqrt(A[i] * x)
    end
    A
end

function lvstd(A::AbstractArray{T, N}, dims::NTuple{M, Int}, corrected::Bool) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        C = hvncat(ntuple(_ -> 1, Val(N)), true, lvstd1(A))
    else
        B = lvmean(A, dims=dims)
        Dᴮ = size(B)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : Dᴮ[d], N)
        Tₒ = Base.promote_op(/, T, Int)
        C = zeros(Tₒ, Dᴮ)
        sumsqdiff!(C, A, B, Dᴮ′)
        Dᴬ = size(A)
        denom = one(Tₒ)
        for d ∈ eachindex(Dᴬ)
            denom = d ∈ dims ? denom * Dᴬ[d] : denom
        end
        denom = corrected ? denom - one(Tₒ) : denom
        x = inv(denom)
        smul_sqrt!(C, x)
    end
    return C
end
lvstd(A::AbstractArray{T, N}, dims::Int, corrected) where {T, N} = lvstd(A, (dims,), corrected)
lvstd(A::AbstractArray{T, N}; dims=:, corrected=true) where {T, N} = lvstd(A, dims, corrected)
lvstd(A::AbstractArray{T, N}, ::Colon, corrected) where {T, N} = lvstd1(A, corrected)

lvstd1(A::AbstractArray{T, N}, corrected::Bool=true) where {T, N} = √(lvvar1(A, corrected))

################ threaded version
function tsmul_sqrt!(A::AbstractArray{T, N}, x::T) where {T, N}
    @tturbo for i ∈ eachindex(A)
        A[i] = sqrt(A[i] * x)
    end
    A
end

function lvtstd(A::AbstractArray{T, N}, dims::NTuple{M, Int}, corrected::Bool) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        C = hvncat(ntuple(_ -> 1, Val(N)), true, lvtstd1(A))
    else
        B = lvmean(A, dims=dims)
        Dᴮ = size(B)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : Dᴮ[d], N)
        Tₒ = Base.promote_op(/, T, Int)
        C = zeros(Tₒ, Dᴮ)
        tsumsqdiff!(C, A, B, Dᴮ′)
        Dᴬ = size(A)
        denom = one(Tₒ)
        for d ∈ eachindex(Dᴬ)
            denom = d ∈ dims ? denom * Dᴬ[d] : denom
        end
        denom = corrected ? denom - one(Tₒ) : denom
        x = inv(denom)
        tsmul_sqrt!(C, x)
    end
    return C
end
lvtstd(A::AbstractArray{T, N}, dims::Int, corrected) where {T, N} = lvtstd(A, (dims,), corrected)
lvtstd(A::AbstractArray{T, N}; dims=:, corrected=true) where {T, N} = lvtstd(A, dims, corrected)
lvtstd(A::AbstractArray{T, N}, ::Colon, corrected) where {T, N} = lvtstd1(A, corrected)

lvtstd1(A::AbstractArray{T, N}, corrected::Bool=true) where {T, N} = √(lvtvar1(A, corrected))


# ################################################################
# # covariance in multiple dimensions

# function outerloopgen(N::Int)
#     loops = Expr(:for)
#     block = Expr(:block)
#     for d = N:-1:1
#         ex = Expr(:(=), Symbol(:j_, d), Expr(:call, :axes, :B, d))
#         push!(block.args, ex)
#     end
#     push!(loops.args, block)
#     loops
# end
# function outerloopgen(N::Int, D)
#     loops = Expr(:for)
#     block = Expr(:block)
#     params = D.parameters
#     for d = N:-1:1
#         if params[d] != Static.One
#             ex = Expr(:(=), Symbol(:j_, d), Expr(:call, :axes, :B, d))
#             push!(block.args, ex)
#         end
#     end
#     push!(loops.args, block)
#     loops
# end

# function sumsqdiffij_body(N::Int, D)
#     body = Expr(:block)
#     params = D.parameters
#     # # As it was originally conceived, but the more efficient form (iᵢ, iⱼ)
#     # # actually confers enhanced clarity.
#     # aᵢ = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
#     # bᵢ = Expr(:ref, :B, ntuple(d -> params[d] == Static.One ? 1 : Symbol(:i_, d), N)...)
#     # aⱼ = Expr(:ref, :A, ntuple(d -> Symbol(:j_, d), N)...)
#     # bⱼ = Expr(:ref, :B, ntuple(d -> params[d] == Static.One ? 1 : Symbol(:j_, d), N)...)
#     # c = Expr(:ref, :C, ntuple(d -> params[d] == Static.One ? 1 : Symbol(:i_, d), N)...,
#     #          ntuple(d -> params[d] == Static.One ? 1 : Symbol(:j_, d), N)...)
#     iᵢ = ntuple(d -> params[d] == Static.One ? 1 : Symbol(:i_, d), N)
#     iⱼ = ntuple(d -> params[d] == Static.One ? 1 : Symbol(:j_, d), N)
#     aᵢ = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
#     bᵢ = Expr(:ref, :B, iᵢ...)
#     aⱼ = Expr(:ref, :A, ntuple(d -> params[d] == Static.One ? Symbol(:i_, d) : Symbol(:j_, d), N)...)
#     bⱼ = Expr(:ref, :B, iⱼ...)
#     c = Expr(:ref, :C, iᵢ..., iⱼ...)
#     δᵢ = Expr(:(=), :Δᵢ, Expr(:call, :-, aᵢ, bᵢ))
#     δⱼ = Expr(:(=), :Δⱼ, Expr(:call, :-, aⱼ, bⱼ))
#     e = Expr(:(=), c, Expr(:call, :+, c, Expr(:call, :*, :Δᵢ, :Δⱼ)))
#     push!(body.args, δᵢ)
#     push!(body.args, δⱼ)
#     push!(body.args, e)
#     body
# end

# function sumsqdiffij_quote(N::Int, D)
#     outer = outerloopgen(N)
#     # outer = outerloopgen(N, D)
#     inner = loopgen(N)
#     body = sumsqdiffij_body(N, D)
#     push!(inner.args, body)
#     push!(outer.args, inner)
#     outer
#     # return quote
#     #     @turbo $outer
#     #     return C
#     # end
# end

# @generated function sumsqdiffij!(C::AbstractArray{Tₒ, Nₒ}, A::AbstractArray{T, N},
#                                  B::AbstractArray{T, N}, dims::D) where {Tₒ, Nₒ, T, N, D}
#     sumsqdiffij_quote(N, D)
# end

# function lvcov(A::AbstractArray{T, N}, dims::NTuple{M, Int}, corrected::Bool) where {T, N, M}
#     B = lvmean(A, dims=dims)
#     Dᴮ = size(B)
#     Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : Dᴮ[d], N)
#     Tₒ = Base.promote_op(/, T, Int)
#     C = zeros(Tₒ, Dᴮ..., Dᴮ...)
#     sumsqdiffij!(C, A, B, Dᴮ′)
#     Dᴬ = size(A)
#     denom = one(Tₒ)
#     for d ∈ eachindex(Dᴬ)
#         denom = d ∈ dims ? denom * Dᴬ[d] : denom
#     end
#     denom = corrected ? denom - one(Tₒ) : denom
#     x = inv(denom)
#     smul!(C, x)
#     C
# end
