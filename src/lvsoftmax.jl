#
# Date created: 2022-03-10
# Author: aradclif
#
#
############################################################################################
# Why not just exp! the result of lvlogsoftmax? Consider that such an approach would perform
# the operations:
# C[i_1,…] = A[i_1,…] - B[i_1, 1, …]
# C[i_1,…] = exp(C[i_1,…])
# One can combine this in order to traverse the memory once, i.e.
# C[i_1,…] = exp(A[i_1,…] - B[i_1, 1, …])

function aminusb_expbody(N::Int, D)
    body = Expr(:block)
    params = D.parameters
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    b = Expr(:ref, :B, ntuple(d -> params[d] == Static.One ? 1 : Symbol(:i_, d), N)...)
    c = Expr(:ref, :C, ntuple(d -> Symbol(:i_, d), N)...)
    e = Expr(:(=), c, Expr(:call, :exp, Expr(:call, :-, a, b)))
    push!(body.args, e)
    body
end

function aminusb_exp_quote(N::Int, D)
    ls = loopgen(N)
    body = aminusb_expbody(N, D)
    push!(ls.args, body)
    return quote
        @turbo $ls
        return C
    end
end

@generated function aminusb_exp!(C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, B::AbstractArray{Tₘ, N}, dims::D) where {Tₒ, T, Tₘ, N, D}
    aminusb_exp_quote(N, D)
end

function _lvsoftmax(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        C = lvsoftmax1(A)
    else
        B = lvlogsumexp(A, dims=dims, multithreaded=false)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(B, d), Val(N))
        C = similar(A, promote_type(T, eltype(B)))
        aminusb_exp!(C, A, B, Dᴮ′)
    end
    return C
end

_lvsoftmax(A::AbstractArray{T, N}, dims::Int) where {T, N} = _lvsoftmax(A, (dims,))
_lvsoftmax(A::AbstractArray{T, N}) where {T, N} = lvsoftmax1(A)
_lvsoftmax(A::AbstractArray{T, N}, ::Colon) where {T, N} = lvsoftmax1(A)

function lvsoftmax1(A::AbstractArray{T, N}) where {T, N}
    b = lvlogsumexp1(A)
    C = similar(A, Base.promote_op(exp, T))
    @turbo for i ∈ eachindex(A)
        C[i] = exp(A[i] - b)
    end
    C
end

################ threaded version

function taminusb_exp_quote(N::Int, D)
    ls = loopgen(N)
    body = aminusb_expbody(N, D)
    push!(ls.args, body)
    return quote
        @tturbo $ls
        return C
    end
end

@generated function taminusb_exp!(C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, B::AbstractArray{Tₘ, N}, dims::D) where {Tₒ, T, Tₘ, N, D}
    taminusb_exp_quote(N, D)
end

function _lvtsoftmax(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        C = lvtsoftmax1(A)
    else
        B = lvlogsumexp(A, dims=dims, multithreaded=true)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(B, d), Val(N))
        C = similar(A, promote_type(T, eltype(B)))
        taminusb_exp!(C, A, B, Dᴮ′)
    end
    return C
end

_lvtsoftmax(A::AbstractArray{T, N}, dims::Int) where {T, N} = _lvtsoftmax(A, (dims,))
_lvtsoftmax(A::AbstractArray{T, N}) where {T, N} = lvtsoftmax1(A)
_lvtsoftmax(A::AbstractArray{T, N}, ::Colon) where {T, N} = lvtsoftmax1(A)

function lvtsoftmax1(A::AbstractArray{T, N}) where {T, N}
    b = lvtlogsumexp1(A)
    C = similar(A, Base.promote_op(exp, T))
    @tturbo for i ∈ eachindex(A)
        C[i] = exp(A[i] - b)
    end
    C
end

################
# Common interface
function lvsoftmax(A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N}
    if (multithreaded === :auto && length(A) > 4095) || multithreaded === true
        α = _lvtsoftmax(A, dims)
    else
        α = _lvsoftmax(A, dims)
    end
    return α
end
