#
# Date created: 2022-03-08
# Author: aradclif
#
#
############################################################################################
# Cleaned version of experimental.jl
using Static
using LoopVectorization

function loopgen(N::Int) #A::AbstractArray{T, N} where {T, N}
    if N == 1
        loops = Expr(:for)
        ex = Expr(:(=), Symbol(:i_, 1), Expr(:call, :axes, :A, 1))
        push!(loops.args, ex)
    else
        loops = Expr(:for)
        block = Expr(:block)
        for d = N:-1:1
            ex = Expr(:(=), Symbol(:i_, d), Expr(:call, :axes, :A, d))
            push!(block.args, ex)
        end
        push!(loops.args, block)
    end
    return loops
end
loopgen(A::AbstractArray{T, N}) where {T, N} = loopgen(N)

function reducebody(f, N, D)
    body = Expr(:block)
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    b = Expr(:ref, :B, ntuple(d -> D[d] == 1 ? 1 : Symbol(:i_, d), N)...)
    e = Expr(:(=), b, Expr(:call, Symbol(f), b, a))
    push!(body.args, e)
    body
end
function mapreducebody(f, op, N, D)
    body = Expr(:block)
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    b = Expr(:ref, :B, ntuple(d -> D[d] == 1 ? 1 : Symbol(:i_, d), N)...)
    e = Expr(:(=), b, Expr(:call, Symbol(op), b, Expr(:call, Symbol(f), a)))
    push!(body.args, e)
    body
end

#### quote for use in @generated
function reduce_quote(F, N::Int, D)
    ls = loopgen(N)
    body = Expr(:block)
    params = D.parameters
    f = F.instance
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    b = Expr(:ref, :B, ntuple(d -> params[d] === Static.One ? 1 : Symbol(:i_, d), N)...)
    e = Expr(:(=), b, Expr(:call, Symbol(f), b, a))
    push!(body.args, e)
    push!(ls.args, body)
    return quote
        @turbo $ls
        return B
    end
end

function _lvreduce(f, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        B = hvncat(ntuple(_ -> 1, Val(N)), true, lvreduce1(f, A))
    else
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), Val(N))
        B = zeros(Base.promote_op(f, T), Dᴮ′)
        _lvreduce!(f, B, A, Dᴮ′)
    end
    return B
end

@generated function _lvreduce!(f::F, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, Tₒ, T, N, D}
    reduce_quote(F, N, D)
end

################
# Handle scalar dims by wrapping in Tuple
_lvreduce(f, A::AbstractArray{T, N}, dims::Int) where {T, N} = _lvreduce(f, A, (dims,))
# Convenience dispatches to match JuliaBase
# lvreduce(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvreduce(f, A, dims)
lvreduce(f, A::AbstractArray{T, N}) where {T, N} = lvreduce1(f, A)
_lvreduce(f, A::AbstractArray{T, N}, ::Colon) where {T, N} = lvreduce1(f, A)

# When dimensions unspecified, treat as vector. Aside from special cases,
# one must usually provide an initial value in order for the reduce to be defined.
# In lieu of a proper approach, zero seems to be a decent guess.
@generated function lvreduce1(f::F, A::AbstractArray{T, N}) where {F, T, N}
    f = F.instance
    quote
        s = zero($T)
        @turbo for i ∈ eachindex(A)
            s = $f(s, A[i])
        end
        s
    end
end
for (op, init) ∈ zip((:+, :-, :*, :max, :min), (:zero, :zero, :one, :typemin, :typemax))
    @eval function _lvreduce(::typeof($op), A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
        if ntuple(identity, Val(N)) ⊆ dims
            B = hvncat(ntuple(_ -> 1, Val(N)), true, lvreduce1($op, A))
        else
            Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), Val(N))
            B = fill($init(T), Dᴮ′)
            _lvreduce!($op, B, A, Dᴮ′)
        end
        return B
    end
    @eval function lvreduce1(::typeof($op), A::AbstractArray{T, N}) where {T, N}
        s = $init(T)
        @turbo for i ∈ eachindex(A)
            s = $op(s, A[i])
        end
        s
    end
end

# # Convenience definitions
# lvsum(A::AbstractArray{T, N}; dims=:) where {T, N} = _lvreduce(+, A, dims=dims)
# lvprod(A::AbstractArray{T, N}; dims=:) where {T, N} = _lvreduce(*, A, dims=dims)
# lvmaximum(A::AbstractArray{T, N}; dims=:) where {T, N} = _lvreduce(max, A, dims=dims)
# lvminimum(A::AbstractArray{T, N}; dims=:) where {T, N} = _lvreduce(min, A, dims=dims)
# lvextrema(A::AbstractArray{T, N}; dims=:) where {T, N} =
#     collect(zip(lvminimum(A, dims=dims), lvmaximum(A, dims=dims)))

################
function _lvreduce_init(f, A::AbstractArray{T, N}, dims::NTuple{M, Int}, init) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        B = hvncat(ntuple(_ -> 1, Val(N)), true, lvreduce1_init(f, A, (init,)))
    else
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), Val(N))
        B = fill(Base.promote_op(f, T)(init), Dᴮ′)
        _lvreduce!(f, B, A, Dᴮ′)
    end
    return B
end
@generated function lvreduce1_init(f::F, A::AbstractArray{T, N}, init) where {F, T, N}
    f = F.instance
    quote
        s = $T(init[1])
        @turbo for i ∈ eachindex(A)
            s = $f(s, A[i])
        end
        s
    end
end
_lvreduce_init(f, A::AbstractArray{T, N}, dims::Int, init) where {T, N} =
    _lvreduce_init(f, A, (dims,), init)
_lvreduce(f, A::AbstractArray{T, N}; dims=:, init=nothing) where {T, N} =
    init === nothing ? _lvreduce(f, A, dims) : _lvreduce_init(f, A, dims, init)

# Handle scalar dims by wrapping in Tuple
_lvreduce_init(f, A::AbstractArray{T, N}, ::Colon, init) where {T, N} = lvreduce1_init(f, A, init)

################
# Common interface for everything related to reduce
function lvreduce(f, A::AbstractArray{T, N}; dims=:, init=nothing, multithreaded=:auto) where {T, N}
    if (multithreaded === :auto && length(A) > 4095) || multithreaded === true
        B = _lvtreduce(f, A, dims=dims, init=init)
    else
        B = _lvreduce(f, A, dims=dims, init=init)
    end
    return B
end

# Convenience definitions
lvsum(A::AbstractArray{T, N}; dims=:, init=nothing, multithreaded=:auto) where {T, N} =
    lvreduce(+, A, dims=dims, init=init, multithreaded=multithreaded)
lvprod(A::AbstractArray{T, N}; dims=:, init=nothing, multithreaded=:auto) where {T, N} =
    lvreduce(*, A, dims=dims, init=init, multithreaded=multithreaded)
lvmaximum(A::AbstractArray{T, N}; dims=:, init=nothing, multithreaded=:auto) where {T, N} =
    lvreduce(max, A, dims=dims, init=init, multithreaded=multithreaded)
lvminimum(A::AbstractArray{T, N}; dims=:, init=nothing, multithreaded=:auto) where {T, N} =
    lvreduce(min, A, dims=dims, init=init, multithreaded=multithreaded)
lvextrema(A::AbstractArray{T, N}; dims=:, init=nothing, multithreaded=:auto) where {T, N} =
    collect(zip(lvminimum(A, dims=dims, init=init, multithreaded=multithreaded),
                lvmaximum(A, dims=dims, init=init, multithreaded=multithreaded)))
################

function mapreduce_quote(F, OP, N::Int, D)
    ls = loopgen(N)
    body = Expr(:block)
    params = D.parameters
    f = F.instance
    op = OP.instance
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    b = Expr(:ref, :B, ntuple(d -> params[d] === Static.One ? 1 : Symbol(:i_, d), N)...)
    e = Expr(:(=), b, Expr(:call, Symbol(op), b, Expr(:call, Symbol(f), a)))
    push!(body.args, e)
    push!(ls.args, body)
    return quote
        @turbo $ls
        return B
    end
end

function lvmapreduce(f, op, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        B = hvncat(ntuple(_ -> 1, Val(N)), true, lvmapreduce1(f, op, A))
    else
        Dᴬ = size(A)
        Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], N)
        B = zeros(Base.promote_op(f, T), Dᴮ)
        # Dᴮ′ = ntuple(d -> StaticInt(Dᴮ[d]), N)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : Dᴮ[d], N)
        _lvmapreduce!(f, op, B, A, Dᴮ′)
    end
    return B
end

@generated function _lvmapreduce!(f::F, op::OP, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, Tₒ, T, N, D}
    mapreduce_quote(F, OP, N, D)
end


################
# Handle scalar dims by wrapping in Tuple
lvmapreduce(f, op, A::AbstractArray{T, N}, dims::Int) where {T, N} = lvmapreduce(f, op, A, (dims,))
# Convenience dispatches to match JuliaBase
lvmapreduce(f, op, A::AbstractArray{T, N}; dims=:) where {T, N} = lvmapreduce(f, op, A, dims)
lvmapreduce(f, op, A::AbstractArray{T, N}) where {T, N} = lvmapreduce1(f, op, A)
lvmapreduce(f, op, A::AbstractArray{T, N}, ::Colon) where {T, N} = lvmapreduce1(f, op, A)


# When dimensions unspecified, treat as vector. Aside from special cases,
# one must usually provide an initial value in order for the reduce to be defined.
# In lieu of a proper approach, zero seems to be a decent guess.
@generated function lvmapreduce1(f::F, op::OP, A::AbstractArray{T, N}) where {F, OP, T, N}
    op = OP.instance
    f = F.instance
    quote
        s = zero(Base.promote_op($f, $T))
        @turbo for i ∈ eachindex(A)
            s = $op(s, $f(A[i]))
        end
        s
    end
end
for (op, init) ∈ zip((:+, :-, :*, :max, :min), (:zero, :zero, :one, :typemin, :typemax))
    @eval function lvmapreduce(f, ::typeof($op), A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
        if ntuple(identity, Val(N)) ⊆ dims
            B = hvncat(ntuple(_ -> 1, Val(N)), true, lvmapreduce1(f, $op, A))
        else
            Dᴬ = size(A)
            Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], N)
            B = fill($init(Base.promote_op(f, T)), Dᴮ)
            # Dᴮ′ = ntuple(d -> StaticInt(Dᴮ[d]), N)
            Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : Dᴮ[d], N)
            _lvmapreduce!(f, $op, B, A, Dᴮ′)
        end
        return B
    end
    @eval function lvmapreduce1(f, ::typeof($op), A::AbstractArray{T, N}) where {T, N}
        s = $init(Base.promote_op(f, T))
        @turbo for i ∈ eachindex(A)
            s = $op(s, f(A[i]))
        end
        s
    end
end

# Convenience definitions
lvsum(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvmapreduce(f, +, A, dims=dims)
lvprod(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvmapreduce(f, *, A, dims=dims)
lvmaximum(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvmapreduce(f, max, A, dims=dims)
lvminimum(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvmapreduce(f, min, A, dims=dims)
lvextrema(f, A::AbstractArray{T, N}; dims=:) where {T, N} =
    collect(zip(lvminimum(f, A, dims=dims), lvmaximum(f, A, dims=dims)))
############################################################################################
#### threaded versions
function treduce_quote(F, N::Int, D)
    ls = loopgen(N)
    body = Expr(:block)
    params = D.parameters
    f = F.instance
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    b = Expr(:ref, :B, ntuple(d -> params[d] === Static.One ? 1 : Symbol(:i_, d), N)...)
    e = Expr(:(=), b, Expr(:call, Symbol(f), b, a))
    push!(body.args, e)
    push!(ls.args, body)
    return quote
        @tturbo $ls
        return B
    end
end
function lvtreduce(f, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        B = hvncat(ntuple(_ -> 1, Val(N)), true, lvtreduce1(f, A))
    else
        Dᴬ = size(A)
        Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], N)
        B = zeros(Base.promote_op(f, T), Dᴮ)
        # Dᴮ′ = ntuple(d -> StaticInt(Dᴮ[d]), N)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : Dᴮ[d], N)
        _lvtreduce!(f, B, A, Dᴮ′)
    end
    return B
end
@generated function _lvtreduce!(f::F, B::AbstractArray{T, N}, A::AbstractArray{T, N}, dims::D) where {F, T, N, D}
    treduce_quote(F, N, D)
end

################
# Handle scalar dims by wrapping in Tuple
lvtreduce(f, A::AbstractArray{T, N}, dims::Int) where {T, N} = lvtreduce(f, A, (dims,))
# Convenience dispatches to match JuliaBase
lvtreduce(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvtreduce(f, A, dims)
lvtreduce(f, A::AbstractArray{T, N}) where {T, N} = lvtreduce1(f, A)
lvtreduce(f, A::AbstractArray{T, N}, ::Colon) where {T, N} = lvtreduce1(f, A)


# When dimensions unspecified, treat as vector. Aside from special cases,
# one must usually provide an initial value in order for the reduce to be defined.
# In lieu of a proper approach, zero seems to be a decent guess.
@generated function lvtreduce1(f::F, A::AbstractArray{T, N}) where {F, T, N}
    f = F.instance
    quote
        s = zero($T)
        @tturbo for i ∈ eachindex(A)
            s = $f(s, A[i])
        end
        s
    end
end
for (op, init) ∈ zip((:+, :-, :*, :max, :min), (:zero, :zero, :one, :typemin, :typemax))
    @eval function lvtreduce(::typeof($op), A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
        if ntuple(identity, Val(N)) ⊆ dims
            B = hvncat(ntuple(_ -> 1, Val(N)), true, lvtreduce1($op, A))
        else
            Dᴬ = size(A)
            Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], N)
            B = fill($init(T), Dᴮ)
            # Dᴮ′ = ntuple(d -> StaticInt(Dᴮ[d]), N)
            Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : Dᴮ[d], N)
            _lvtreduce!($op, B, A, Dᴮ′)
        end
        return B
    end
    @eval function lvtreduce1(::typeof($op), A::AbstractArray{T, N}) where {T, N}
        s = $init(T)
        @tturbo for i ∈ eachindex(A)
            s = $op(s, A[i])
        end
        s
    end
end

# Convenience definitions
lvtsum(A::AbstractArray{T, N}; dims=:) where {T, N} = lvtreduce(+, A, dims=dims)
lvtprod(A::AbstractArray{T, N}; dims=:) where {T, N} = lvtreduce(*, A, dims=dims)
lvtmaximum(A::AbstractArray{T, N}; dims=:) where {T, N} = lvtreduce(max, A, dims=dims)
lvtminimum(A::AbstractArray{T, N}; dims=:) where {T, N} = lvtreduce(min, A, dims=dims)
lvtextrema(A::AbstractArray{T, N}; dims=:) where {T, N} =
    collect(zip(lvtminimum(A, dims=dims), lvtmaximum(A, dims=dims)))
################


function tmapreduce_quote(F, OP, N::Int, D)
    ls = loopgen(N)
    body = Expr(:block)
    params = D.parameters
    f = F.instance
    op = OP.instance
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    b = Expr(:ref, :B, ntuple(d -> params[d] === Static.One ? 1 : Symbol(:i_, d), N)...)
    e = Expr(:(=), b, Expr(:call, Symbol(op), b, Expr(:call, Symbol(f), a)))
    push!(body.args, e)
    push!(ls.args, body)
    return quote
        @tturbo $ls
        return B
    end
end
function lvtmapreduce(f, op, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        B = hvncat(ntuple(_ -> 1, Val(N)), true, lvtmapreduce1(f, op, A))
    else
        Dᴬ = size(A)
        Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], N)
        B = zeros(Base.promote_op(f, T), Dᴮ)
        # Dᴮ′ = ntuple(d -> StaticInt(Dᴮ[d]), N)
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : Dᴮ[d], N)
        _lvtmapreduce!(f, op, B, A, Dᴮ′)
    end
    return B
end

@generated function _lvtmapreduce!(f::F, op::OP, B::AbstractArray{T, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, T, N, D}
    tmapreduce_quote(F, OP, N, D)
end

################
# Handle scalar dims by wrapping in Tuple
lvtmapreduce(f, op, A::AbstractArray{T, N}, dims::Int) where {T, N} = lvtmapreduce(f, op, A, (dims,))
# Convenience dispatches to match JuliaBase
lvtmapreduce(f, op, A::AbstractArray{T, N}; dims=:) where {T, N} = lvtmapreduce(f, op, A, dims)
lvtmapreduce(f, op, A::AbstractArray{T, N}) where {T, N} = lvtmapreduce1(f, op, A)
lvtmapreduce(f, op, A::AbstractArray{T, N}, ::Colon) where {T, N} = lvtmapreduce1(f, op, A)


# When dimensions unspecified, treat as vector. Aside from special cases,
# one must usually provide an initial value in order for the reduce to be defined.
# In lieu of a proper approach, zero seems to be a decent guess.
@generated function lvtmapreduce1(f::F, op::OP, A::AbstractArray{T, N}) where {F, OP, T, N}
    f = F.instance
    op = OP.instance
    quote
        s = zero(Base.promote_op($f, $T))
        @tturbo for i ∈ eachindex(A)
            s = $op(s, $f(A[i]))
        end
        s
    end
end
for (op, init) ∈ zip((:+, :-, :*, :max, :min), (:zero, :zero, :one, :typemin, :typemax))
    @eval function lvtmapreduce(f, ::typeof($op), A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
        if ntuple(identity, Val(N)) ⊆ dims
            B = hvncat(ntuple(_ -> 1, Val(N)), true, lvtmapreduce1(f, $op, A))
        else
            Dᴬ = size(A)
            Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], N)
            B = fill($init(Base.promote_op(f, T)), Dᴮ)
            # Dᴮ′ = ntuple(d -> StaticInt(Dᴮ[d]), N)
            Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : Dᴮ[d], N)
            _lvtmapreduce!(f, $op, B, A, Dᴮ′)
        end
        return B
    end
    @eval function lvtmapreduce1(f, ::typeof($op), A::AbstractArray{T, N}) where {T, N}
        s = $init(Base.promote_op(f, T))
        @tturbo for i ∈ eachindex(A)
            s = $op(s, f(A[i]))
        end
        s
    end
end

# Convenience definitions
lvtsum(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvtmapreduce(f, +, A, dims=dims)
lvtprod(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvtmapreduce(f, *, A, dims=dims)
lvtmaximum(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvtmapreduce(f, max, A, dims=dims)
lvtminimum(f, A::AbstractArray{T, N}; dims=:) where {T, N} = lvtmapreduce(f, min, A, dims=dims)
lvtextrema(f, A::AbstractArray{T, N}; dims=:) where {T, N} =
    collect(zip(lvtminimum(f, A, dims=dims), lvtmaximum(f, A, dims=dims)))
################
