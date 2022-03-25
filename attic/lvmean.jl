#
# Date created: 2022-03-09
# Author: aradclif
#
#
############################################################################################
# Building on lvreduce.jl

# function smul_inplacebody(N::Int)
#     body = Expr(:block)
#     a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
#     e = Expr(:(=), a, Expr(:call, :*, a, :x))
#     push!(body.args, e)
#     body
# end

function transform_inplacebody(f, N::Int)
    body = Expr(:block)
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    e = Expr(:(=), a, Expr(:call, Symbol(f), a))
    push!(body.args, e)
    body
end

# function smul_inplace_quote(N::Int)
#     ls = loopgen(N)
#     body = smul_inplacebody(N)
#     push!(ls.args, body)
#     return quote
#         @turbo $ls
#         return A
#     end
# end
# @generated function smul!(A::AbstractArray{T, N}, x::T) where {T, N}
#     smul_inplace_quote(N)
# end

function _smul!(A::AbstractArray{T, N}, x::T) where {T, N}
    @turbo for i ∈ eachindex(A)
        A[i] *= x
    end
    A
end

function _smul!(B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, x::Tₒ) where {Tₒ, T, N}
    @turbo for i ∈ eachindex(A)
        B[i] = A[i] * x
    end
    B
end
_smul(A::AbstractArray{T, N}, x::Tₒ) where {Tₒ, T, N} = _smul!(similar(A, Tₒ), A, x)

function _lvmean(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        return hvncat(ntuple(_ -> 1, Val(N)), true, lvmean1(A))
    else
        B = lvsum(A, dims=dims, multithreaded=false)
        denom = 1
        for d = 1:N
            denom = d ∈ dims ? denom * size(A, d) : denom
        end
        x = inv(denom)
        return eltype(B) <: AbstractFloat ? _smul!(B, x) : _smul(B, x)
    end
end
_lvmean(A::AbstractArray{T, N}, dims::Int) where {T, N} = _lvmean(A, (dims,))
_lvmean(A::AbstractArray{T, N}) where {T, N} = lvmean1(A)
_lvmean(A::AbstractArray{T, N}, ::Colon) where {T, N} = lvmean1(A)

function lvmean1(A::AbstractArray{T, N}) where {T, N}
    s = zero(Base.promote_op(+, T, Int))
    @turbo for i ∈ eachindex(A)
        s += A[i]
    end
    s / length(A)
end

################ the mapreduce version analogous to Julia base Statistics
# Although, truly, why anyone would _only_ compute mean of a transformation is
# puzzling. Consequently, from a theoretical and practical perspective, one
# would usually want to transform the samples...
function _lvmean(f, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        return hvncat(ntuple(_ -> 1, Val(N)), true, lvmean1(f, A))
    else
        B = lvsum(f, A, dims=dims, multithreaded=false)
        denom = 1
        for d = 1:N
            denom = d ∈ dims ? denom * size(A, d) : denom
        end
        x = inv(denom)
        return eltype(B) <: AbstractFloat ? _smul!(B, x) : _smul(B, x)
    end
end
_lvmean(f, A::AbstractArray{T, N}, dims::Int) where {T, N} = _lvmean(f, A, (dims,))
_lvmean(f, A::AbstractArray{T, N}) where {T, N} = lvmean1(f, A)
_lvmean(f, A::AbstractArray{T, N}, ::Colon) where {T, N} = lvmean1(f, A)

@generated function lvmean1(f::F, A::AbstractArray{T, N}) where {F, T, N}
    f = F.instance
    Tₒ = Base.promote_op(+, Base.promote_op(f, T), Int)
    quote
        s = zero($Tₒ)
        @turbo for i ∈ eachindex(A)
            s += $f(A[i])
        end
        s / length(A)
    end
end

################ threaded version

function _tsmul!(A::AbstractArray{T, N}, x::T) where {T, N}
    @tturbo for i ∈ eachindex(A)
        A[i] *= x
    end
    A
end

function _tsmul!(B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, x::Tₒ) where {Tₒ, T, N}
    @tturbo for i ∈ eachindex(A)
        B[i] = A[i] * x
    end
    B
end
_tsmul(A::AbstractArray{T, N}, x::Tₒ) where {Tₒ, T, N} = _tsmul!(similar(A, Tₒ), A, x)

function _lvtmean(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        return hvncat(ntuple(_ -> 1, Val(N)), true, lvtmean1(A))
    else
        B = lvsum(A, dims=dims, multithreaded=true)
        denom = 1
        for d = 1:N
            denom = d ∈ dims ? denom * size(A, d) : denom
        end
        x = inv(denom)
        return eltype(B) <: AbstractFloat ? _tsmul!(B, x) : _tsmul(B, x)
    end
end
_lvtmean(A::AbstractArray{T, N}, dims::Int) where {T, N} = _lvtmean(A, (dims,))
_lvtmean(A::AbstractArray{T, N}) where {T, N} = lvtmean1(A)
_lvtmean(A::AbstractArray{T, N}, ::Colon) where {T, N} = lvtmean1(A)

function lvtmean1(A::AbstractArray{T, N}) where {T, N}
    s = zero(Base.promote_op(+, T, Int))
    @tturbo for i ∈ eachindex(A)
        s += A[i]
    end
    s / length(A)
end

################
function _lvtmean(f, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        return hvncat(ntuple(_ -> 1, Val(N)), true, lvtmean1(f, A))
    else
        B = lvsum(f, A, dims=dims, multithreaded=true)
        denom = 1
        for d = 1:N
            denom = d ∈ dims ? denom * size(A, d) : denom
        end
        x = inv(denom)
        return eltype(B) <: AbstractFloat ? _tsmul!(B, x) : _tsmul(B, x)
    end
end
_lvtmean(f, A::AbstractArray{T, N}, dims::Int) where {T, N} = _lvtmean(f, A, (dims,))
_lvtmean(f, A::AbstractArray{T, N}) where {T, N} = _lvtmean1(f, A)
_lvtmean(f, A::AbstractArray{T, N}, ::Colon) where {T, N} = _lvtmean1(f, A)

@generated function _lvtmean1(f::F, A::AbstractArray{T, N}) where {F, T, N}
    f = F.instance
    Tₒ = Base.promote_op(+, Base.promote_op(f, T), Int)
    quote
        s = zero($Tₒ)
        @tturbo for i ∈ eachindex(A)
            s += $f(A[i])
        end
        s / length(A)
    end
end

################
# Common interface
function lvmean(A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N}
    if (multithreaded === :auto && length(A) > 4095) || multithreaded === true
        μ = _lvtmean(A, dims)
    else
        μ = _lvmean(A, dims)
    end
    return μ
end

function lvmean(f, A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N}
    if (multithreaded === :auto && length(A) > 4095) || multithreaded === true
        μ = _lvtmean(f, A, dims)
    else
        μ = _lvmean(f, A, dims)
    end
    return μ
end
