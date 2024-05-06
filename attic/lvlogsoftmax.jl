#
# Date created: 2022-03-10
# Author: aradclif
#
#
############################################################################################

function aminusbbody(N::Int, D)
    body = Expr(:block)
    params = D.parameters
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    b = Expr(:ref, :B, ntuple(d -> params[d] === Static.One ? 1 : Symbol(:i_, d), N)...)
    c = Expr(:ref, :C, ntuple(d -> Symbol(:i_, d), N)...)
    e = Expr(:(=), c, Expr(:call, :-, a, b))
    push!(body.args, e)
    body
end

function aminusb_quote(N::Int, D)
    ls = loopgen(N)
    body = aminusbbody(N, D)
    push!(ls.args, body)
    return quote
        @turbo check_empty=true $ls
        return C
    end
end

@generated function aminusb!(C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, B::AbstractArray{Tₘ, N},
                             dims::D) where {Tₒ, T, Tₘ, N, D}
    aminusb_quote(N, D)
end

function exp!(A::AbstractArray{T, N}) where {T, N}
    @turbo check_empty=true for i ∈ eachindex(A)
        A[i] = exp(A[i])
    end
    A
end

function _lvlogsoftmax(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        C = lvlogsoftmax1(A)
    else
        B = lvlogsumexp(A, dims=dims, multithreaded=false)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(B, d), Val(N))
        C = similar(A, promote_type(T, eltype(B)))
        aminusb!(C, A, B, Dᴮ′)
    end
    return C
end
_lvlogsoftmax(A::AbstractArray{T, N}, dims::Int) where {T, N} = _lvlogsoftmax(A, (dims,))
_lvlogsoftmax(A::AbstractArray{T, N}) where {T, N} = lvlogsoftmax1(A)
_lvlogsoftmax(A::AbstractArray{T, N}, ::Colon) where {T, N} = lvlogsoftmax1(A)

function lvlogsoftmax1(A::AbstractArray{T, N}) where {T, N}
    b = lvlogsumexp1(A)
    C = similar(A, Base.promote_op(exp, T))
    @turbo check_empty=true for i ∈ eachindex(A)
        C[i] = A[i] - b
    end
    C
end

################ threaded version

function taminusb_quote(N::Int, D)
    ls = loopgen(N)
    body = aminusbbody(N, D)
    push!(ls.args, body)
    return quote
        @tturbo check_empty=true $ls
        return C
    end
end
@generated function taminusb!(C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, B::AbstractArray{Tₘ, N}, dims::D) where {Tₒ, T, Tₘ, N, D}
    taminusb_quote(N, D)
end

function _lvtlogsoftmax(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        C = lvtlogsoftmax1(A)
    else
        B = lvlogsumexp(A, dims=dims, multithreaded=true)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(B, d), Val(N))
        C = similar(A, promote_type(T, eltype(B)))
        taminusb!(C, A, B, Dᴮ′)
    end
    return C
end
_lvtlogsoftmax(A::AbstractArray{T, N}, dims::Int) where {T, N} = _lvtlogsoftmax(A, (dims,))
_lvtlogsoftmax(A::AbstractArray{T, N}) where {T, N} = lvtlogsoftmax1(A)
_lvtlogsoftmax(A::AbstractArray{T, N}, ::Colon) where {T, N} = lvtlogsoftmax1(A)

function lvtlogsoftmax1(A::AbstractArray{T, N}) where {T, N}
    b = lvtlogsumexp1(A)
    C = similar(A, Base.promote_op(exp, T))
    @tturbo check_empty=true for i ∈ eachindex(A)
        C[i] = A[i] - b
    end
    C
end

################
# Common interface
function lvlogsoftmax(A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N}
    if (multithreaded === :auto && length(A) > 4095) || multithreaded === true
        α = _lvtlogsoftmax(A, dims)
    else
        α = _lvlogsoftmax(A, dims)
    end
    return α
end
