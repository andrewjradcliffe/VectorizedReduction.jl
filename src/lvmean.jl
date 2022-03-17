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

function smul!(A::AbstractArray{T, N}, x::T) where {T, N}
    @turbo for i ∈ eachindex(A)
        A[i] *= x
    end
    A
end

function smul!(B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, x::Tₒ) where {Tₒ, T, N}
    @turbo for i ∈ eachindex(A)
        B[i] = A[i] * x
    end
    B
end
smul(A::AbstractArray{T, N}, x::Tₒ) where {Tₒ, T, N} = smul!(similar(A, Tₒ), A, x)

function lvmean(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        return hvncat(ntuple(i -> 1, Val(N)), true, lvmean1(A))
    else
        B = lvsum(A, dims=dims)
        Dᴬ = size(A)
        denom = 1
        for d ∈ eachindex(Dᴬ)
            denom = d ∈ dims ? denom * Dᴬ[d] : denom
        end
        x = inv(denom)
        # smul!(B, x)
        # B
        return eltype(B) <: AbstractFloat ? smul!(B, x) : smul(B, x)
    end
end
lvmean(A::AbstractArray{T, N}, dims::Int) where {T, N} = lvmean(A, (dims,))
lvmean(A::AbstractArray{T, N}; dims=:) where {T, N} = lvmean(A, dims)
lvmean(A::AbstractArray{T, N}) where {T, N} = lvmean1(A)
lvmean(A::AbstractArray{T, N}, ::Colon) where {T, N} = lvmean1(A)

function lvmean1(A::AbstractArray{T, N}) where {T, N}
    s = zero(Base.promote_op(+, T, Int))
    @turbo for i ∈ eachindex(A)
        s += A[i]
    end
    s / length(A)
end

################ the mapreduce version analogous to Julia base Statistics
function lvmean(f, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        return hvncat(ntuple(i -> 1, Val(N)), true, lvmean1(f, A))
    else
        B = lvsum(f, A, dims=dims)
        Dᴬ = size(A)
        denom = 1
        for d ∈ eachindex(Dᴬ)
            denom = d ∈ dims ? denom * Dᴬ[d] : denom
        end
        x = inv(denom)
        # smul!(B, x)
        # B
        return eltype(B) <: AbstractFloat ? smul!(B, x) : smul(B, x)
    end
end
lvmean(f, A::AbstractArray{T, N}, dims::Int) where {T, N} = lvmean(f, A, (dims,))
lvmean(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvmean(f, A, dims)
lvmean(f, A::AbstractArray{T, N}) where {T, N} = lvmean1(f, A)
lvmean(f, A::AbstractArray{T, N}, ::Colon) where {T, N} = lvmean1(f, A)

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

function tsmul!(A::AbstractArray{T, N}, x::T) where {T, N}
    @tturbo for i ∈ eachindex(A)
        A[i] *= x
    end
    A
end

function tsmul!(B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, x::Tₒ) where {Tₒ, T, N}
    @tturbo for i ∈ eachindex(A)
        B[i] = A[i] * x
    end
    B
end
tsmul(A::AbstractArray{T, N}, x::Tₒ) where {Tₒ, T, N} = tsmul!(similar(A, Tₒ), A, x)

function lvtmean(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        return hvncat(ntuple(i -> 1, Val(N)), true, lvtmean1(A))
    else
        B = lvtsum(A, dims=dims)
        Dᴬ = size(A)
        denom = 1
        for d ∈ eachindex(Dᴬ)
            denom = d ∈ dims ? denom * Dᴬ[d] : denom
        end
        x = inv(denom)
        return eltype(B) <: AbstractFloat ? tsmul!(B, x) : tsmul(B, x)
    end
end
lvtmean(A::AbstractArray{T, N}, dims::Int) where {T, N} = lvtmean(A, (dims,))
lvtmean(A::AbstractArray{T, N}; dims=:) where {T, N} = lvtmean(A, dims)
lvtmean(A::AbstractArray{T, N}) where {T, N} = lvtmean1(A)
lvtmean(A::AbstractArray{T, N}, ::Colon) where {T, N} = lvtmean1(A)

function lvtmean1(A::AbstractArray{T, N}) where {T, N}
    s = zero(Base.promote_op(+, T, Int))
    @tturbo for i ∈ eachindex(A)
        s += A[i]
    end
    s / length(A)
end

################
function lvtmean(f, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        return hvncat(ntuple(i -> 1, Val(N)), true, lvtmean1(f, A))
    else
        B = lvtsum(f, A, dims=dims)
        Dᴬ = size(A)
        denom = 1
        for d ∈ eachindex(Dᴬ)
            denom = d ∈ dims ? denom * Dᴬ[d] : denom
        end
        x = inv(denom)
        return eltype(B) <: AbstractFloat ? tsmul!(B, x) : tsmul(B, x)
    end
end
lvtmean(f, A::AbstractArray{T, N}, dims::Int) where {T, N} = lvtmean(f, A, (dims,))
lvtmean(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvtmean(f, A, dims)
lvtmean(f, A::AbstractArray{T, N}) where {T, N} = lvtmean1(f, A)
lvtmean(f, A::AbstractArray{T, N}, ::Colon) where {T, N} = lvtmean1(f, A)

@generated function lvtmean1(f::F, A::AbstractArray{T, N}) where {F, T, N}
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
