#
# Date created: 2022-03-22
# Author: aradclif
#
#
############################################################################################
# - Implementing reduce(op, ...) as simply mapreduce(identity, op, ...)
# - Using compile-time branch resolution rather than dispatch system (well,
#   less on the dispatch system)
_dim(::Type{StaticInt{N}}) where {N} = N::Int
# Stable return type version, otherwise it's just a suggestion
# _dim(::Type{StaticInt{N}})::Int where {N} = N

# Demonstrated: that technically, `f`, can be anonymous. The reduction still needs
# to be a regular binary operation which is known to LoopVectorization.
# Hence, it only seems worthwhile to not force specialization on f.
# Moreover, it is always a slight advantage to use a named function over an anonymous...
# In fact, having changed it over, it is a beautiful thing -- same behavior as Julia Base.

"""
    vvmapreduce(f, op, init, A::AbstractArray, dims=:)

Apply function `f` to each element of `A`, then reduce the result along the dimensions
`dims` using the binary function `op`. The reduction necessitates an initial value `init`
which may be `<:Number` or a function which accepts a single type argument (e.g. `zero`);
`init` is optional for binary operators `+`, `*`, `min`, and `max`.
`dims` may be `::Int`, `::NTuple{M, Int} where {M}` or `::Colon`.

See also: [`vvsum`](@ref), [`vvprod`](@ref), [`vvminimum`](@ref), [`vvmaximum`](@ref)
"""
function vvmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(op, Base.promote_op(f, T), Int), Dᴮ′)
    _vvmapreduce!(f, op, init, B, A, dims)
    return B
end
vvmapreduce(f, op, init, A, dims::Int) = vvmapreduce(f, op, init, A, (dims,))

# dims determination would ideally be non-allocating. Also, who would
# call this anyway? Almost assuredly, a caller would already know dims, hence
# just call _vvmapreduce! anyway.
function vvmapreduce!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}) where {F, OP, I, Tₒ, T, N}
    Dᴬ = size(A)
    Dᴮ = size(B)
    dims = Tuple((d for d ∈ eachindex(Dᴮ) if isone(Dᴮ[d])))
    all(d -> Dᴮ[d] == Dᴬ[d], (d for d ∈ eachindex(Dᴮ) if !isone(Dᴮ[d]))) || throw(DimensionMismatch)
    _vvmapreduce!(f, op, init, B, A, dims)
    return B
end

# Convenience definitions
"""
    vvsum(f, A::AbstractArray, dims=:)

Sum the results of calling `f` on each element of `A` over the specified `dims`.
"""
vvsum(f::F, A, dims) where {F} = vvmapreduce(f, +, zero, A, dims)

"""
    vvprod(f, A::AbstractArray, dims=:)

Multiply the results of calling `f` on each element of `A` over the specified `dims`.
"""
vvprod(f::F, A, dims) where {F} = vvmapreduce(f, *, one, A, dims)

"""
    vvmaximum(f, A::AbstractArray, dims=:)

Compute the maximum value by calling `f` on each element of `A` over the specified `dims`.
"""
vvmaximum(f::F, A, dims) where {F} = vvmapreduce(f, max, typemin, A, dims)

"""
    vvminimum(f, A::AbstractArray, dims=:)

Compute the minimum value by calling `f` on each element of `A` over the specified `dims`.
"""
vvminimum(f::F, A, dims) where {F} = vvmapreduce(f, min, typemax, A, dims)

# ::AbstractArray required in order for kwargs interface to work
"""
    vvsum(A::AbstractArray, dims=:)

Sum the elements of `A` over the specified `dims`.
"""
vvsum(A::AbstractArray, dims) = vvmapreduce(identity, +, zero, A, dims)

"""
    vvprod(A::AbstractArray, dims=:)

Multiply the elements of `A` over the specified `dims`.
"""
vvprod(A::AbstractArray, dims) = vvmapreduce(identity, *, one, A, dims)

"""
    vvmaximum(A::AbstractArray, dims=:)

Compute the maximum value of `A` over the specified `dims`.
"""
vvmaximum(A::AbstractArray, dims) = vvmapreduce(identity, max, typemin, A, dims)

"""
    vvminimum(A::AbstractArray, dims=:)

Compute the minimum value of `A` over the specified `dims`.
"""
vvminimum(A::AbstractArray, dims) = vvmapreduce(identity, min, typemax, A, dims)


# The dispatch on function type is faster if the function is named,
# slower (≈ 15%) for anonymous. This would only affect performance in the REPL,
# hence, it's not really an issue.
vvsum(f::F, A) where {F<:Function} = vvmapreduce(f, +, zero, A, :)
vvprod(f::F, A) where {F<:Function} = vvmapreduce(f, *, one, A, :)
vvmaximum(f::F, A) where {F<:Function} = vvmapreduce(f, max, typemin, A, :)
vvminimum(f::F, A) where {F<:Function} = vvmapreduce(f, min, typemax, A, :)

# ::AbstractArray required in order for kwargs interface to work
vvsum(A::AbstractArray) = vvmapreduce(identity, +, zero, A, :)
vvprod(A::AbstractArray) = vvmapreduce(identity, *, one, A, :)
vvmaximum(A::AbstractArray) = vvmapreduce(identity, max, typemin, A, :)
vvminimum(A::AbstractArray) = vvmapreduce(identity, min, typemax, A, :)

# a custom implementation of extrema is not really worth it, as the time/memory
# cost is approximately the same. Also, it suffers from first dimension reduction error.

"""
    vvextrema(f, A::AbstractArray, dims=:)

Compute the minimum and maximum values by calling `f`  on each element of of `A`
over the specified `dims`.
"""
vvextrema(f::F, A, dims) where {F} = collect(zip(vvminimum(f, A, dims), vvmaximum(f, A, dims)))
vvextrema(f::F, A, ::Colon) where {F} = (vvminimum(f, A, :), vvmaximum(f, A, :))
vvextrema(f::F, A) where {F<:Function} = vvextrema(f, A, :)
# ::AbstractArray required in order for kwargs interface to work

"""
    vvextrema(A::AbstractArray, dims=:)

Compute the minimum and maximum values of `A` over the specified `dims`.
"""
vvextrema(A::AbstractArray, dims) = vvextrema(identity, A, dims)
vvextrema(A::AbstractArray) = (vvminimum(A), vvmaximum(A))

# Define reduce
"""
    vvreduce(op, init, A::AbstractArray, dims=:)

Reduce `A` along the dimensions `dims` using the binary function `op`.
See `vvmapreduce` for description of `op`, `init`, `dims`.

See also: [`vvsum`](@ref), [`vvprod`](@ref), [`vvminimum`](@ref), [`vvmaximum`](@ref)
"""
vvreduce(op::OP, init::I, A, dims) where {OP, I} = vvmapreduce(identity, op, init, A, dims)
vvreduce(op::OP, init::I, A) where {OP, I} = vvmapreduce(identity, op, init, A, :)

for (op, init) ∈ zip((:+, :*, :max,:min), (:zero, :one, :typemin, :typemax))
    @eval vvreduce(::typeof($op), A; dims=:, init=$init) = vvmapreduce(identity, $op, init, A, dims)
    # 2-argument version for common binary ops, but one should just use sum,prod,maximum, minimum
    @eval vvreduce(::typeof($op), A::AbstractArray) = vvmapreduce(identity, $op, $init, A, :)
end

# Provide inherently inefficient kwargs interface. Requires ::AbstractArray in the locations
# indicated above.
"""
    vvmapreduce(f, op, A; dims=:, init)

Identical to non-keyword args version; slightly less performant due to use of kwargs.
"""
vvmapreduce(f, op, A; dims=:, init) = vvmapreduce(f, op, init, A, dims)

"""
    vvreduce(op, A; dims=:, init)

Identical to non-keyword args version; slightly less performant due to use of kwargs.
"""
vvreduce(op, A; dims=:, init) = vvreduce(op, init, A, dims)

"""
    vvsum(f, A; dims=:, init=zero)

Sum the results of calling `f` on each element of `A` over the specified `dims`,
with the sum initialized by `init`, which may be a value `<:Number`
or a function which accepts a single type argument.
"""
vvsum(f, A; dims=:, init=zero) = vvmapreduce(f, +, init, A, dims)

"""
    vvsum(A; dims=:, init=zero)

Sum the elements of `A` over the specified `dims`, with the sum initialized by `init`.
"""
vvsum(A; dims=:, init=zero) = vvmapreduce(identity, +, init, A, dims)

"""
    vvprod(f, A; dims=:, init=one)

Multiply the results of calling `f` on each element of `A` over the specified `dims`,
with the product initialized by `init`, which may be a value `<:Number`
or a function which accepts a single type argument.
"""
vvprod(f, A; dims=:, init=one) = vvmapreduce(f, *, init, A, dims)

"""
    vvprod(A; dims=:, init=one)

Multiply the elements of `A` over the specified `dims`, with the product initialized by `init`.
"""
vvprod(A; dims=:, init=one) = vvmapreduce(identity, *, init, A, dims)

"""
    vvmaximum(f, A; dims=:, init=typemin)

Compute the maximum value of calling `f` on each element of `A` over the specified `dims`,
with the max initialized by `init`, which may be a value `<:Number`
or a function which accepts a single type argument.
"""
vvmaximum(f, A; dims=:, init=typemin) = vvmapreduce(f, max, init, A, dims)

"""
    vvmaximum(A; dims=:, init=typemin)

Compute the maximum value of `A` over the specified `dims`, with the max initialized by `init`.
"""
vvmaximum(A; dims=:, init=typemin) = vvmapreduce(identity, max, init, A, dims)

"""
    vvminimum(f, A; dims=:, init=typemax)

Compute the minimum value of calling `f` on each element of `A` over the specified `dims`,
with the min initialized by `init`, which may be a value `<:Number`
or a function which accepts a single type argument.
"""
vvminimum(f, A; dims=:, init=typemax) = vvmapreduce(f, min, init, A, dims)

"""
    vvminimum(A; dims=:, init=typemax)

Compute the minimum value of `A` over the specified `dims`, with the min initialized by `init`.
"""
vvminimum(A; dims=:, init=typemax) = vvmapreduce(identity, min, init, A, dims)

"""
    vvextrema(f, A::AbstractArray; dims=:, init=(typemax, typemin))

Compute the minimum and maximum values by calling `f`  on each element of of `A`
over the specified `dims`, with the min and max initialized by the respective arguments
of the 2-tuple `init`, which can be any combination of values `<:Number` or functions
which accept a single type argument.
"""
vvextrema(f, A; dims=:, init=(typemax, typemin)) =
    collect(zip(vvmapreduce(f, min, init[1], A, dims), vvmapreduce(f, max, init[2], A, dims)))

"""
    vvextrema(A::AbstractArray; dims=:, init=(typemax, typemin))

Compute the minimum and maximum values of `A` over the specified `dims`,
with the min and max initialized by `init`.
"""
vvextrema(A; dims=:, init=(typemax, typemin)) =
    collect(zip(vvmapreduce(identity, min, init[1], A, dims), vvmapreduce(identity, max, init[2], A, dims)))


# reduction over all dims
@generated function vvmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I, T, N}
    # fsym = F.instance
    opsym = OP.instance
    initsym = I.instance
    # Tₒ = Base.promote_op(opsym, Base.promote_op(fsym, T), Int)
    quote
        # ξ = $initsym($Tₒ)
        ξ = $initsym(Base.promote_op($opsym, Base.promote_op(f, $T), Int))
        @turbo for i ∈ eachindex(A)
            ξ = $opsym(f(A[i]), ξ)
        end
        return ξ
    end
end
vvmapreduce(f::F, op::OP, init::I, A) where {F, OP, I} = vvmapreduce(f, op, init, A, :)

# mixed dimensions reduction

function staticdim_mapreduce_quote(OP, I, static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if d ∈ static_dims
            push!(Bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(rinds, d)
        else
            push!(Bᵥ.args, :)
            push!(nrinds, d)
            push!(Bᵥ′.args, Symbol(:i_, d))
        end
    end
    reverse!(rinds)
    reverse!(nrinds)
    if !isempty(nrinds)
        block = Expr(:block)
        loops = Expr(:for, Expr(:(=), Symbol(:i_, nrinds[1]),
                                Expr(:call, :indices, Expr(:tuple, :A, :B), nrinds[1])), block)
        # loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B), $(nrinds[1]))), block)
        for i = 2:length(nrinds)
            # for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args,
                  Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
                                  Expr(:call, :indices, Expr(:tuple, :A, :B), nrinds[i])), newblock))
            # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :axes, :A, d)), newblock))
            # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            Bᵥ = $Bᵥ
            @turbo $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, Expr(:(=), Symbol(:i_, rinds[1]), Expr(:call, :axes, :A, rinds[1])), block)
        # loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for i = 2:length(rinds)
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                                              Expr(:call, :axes, :A, rinds[i])), newblock))
            # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        return quote
            Bᵥ = $Bᵥ
            $ξ
            @turbo $loops
            Bᵥ[] = ξ
            return B
        end
    end
end

function branches_mapreduce_quote(OP, I, N::Int, M::Int, D)
    static_dims = Int[]
    for m ∈ 1:M
        param = D.parameters[m]
        if param <: StaticInt
            new_dim = _dim(param)::Int
            push!(static_dims, new_dim)
        else
            # tuple of static dimensions
            t = Expr(:tuple)
            for n ∈ static_dims
                push!(t.args, :(StaticInt{$n}()))
            end
            q = Expr(:block, :(dimm = dims[$m]))
            qold = q
            # if-elseif statements
            ifsym = :if
            for n ∈ 1:N
                n ∈ static_dims && continue
                tc = copy(t)
                push!(tc.args, :(StaticInt{$n}()))
                qnew = Expr(ifsym, :(dimm == $n), :(return _vvmapreduce!(f, op, init, B, A, $tc)))
                for r ∈ m+1:M
                    push!(tc.args, :(dims[$r]))
                end
                push!(qold.args, qnew)
                qold = qnew
                ifsym = :elseif
            end
            # else statement
            tc = copy(t)
            for r ∈ m+1:M
                push!(tc.args, :(dims[$r]))
            end
            push!(qold.args, Expr(:block, :(return _vvmapreduce!(f, op, init, B, A, $tc))))
            return q
        end
    end
    return staticdim_mapreduce_quote(OP, I, static_dims, N)
end

@generated function _vvmapreduce!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_mapreduce_quote(OP, I, N, M, D)
end

# This could likely be handled by eachindex, but for completeness:
# this is the case of mapreduce on a single array when rinds = ∅
function map_quote() #N::Int
    # A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    # block = Expr(:block)
    # loops = Expr(:for, Expr(:(=), Symbol(:i_, N), Expr(:call, :indices, Expr(:tuple, :A, :B), N)), block)
    # for d = N-1:-1:1
    #     newblock = Expr(:block)
    #     push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :indices, Expr(:tuple, :A, :B), d)), newblock))
    #     block = newblock
    # end
    # # Push to inside innermost loop
    # setb = Expr(:(=), Expr(:ref, :B, ntuple(d -> Symbol(:i_, d), N)...), Expr(:call, :f, A))
    # push!(block.args, setb)
    # Version using eachindex
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :A, :B)), block)
    setb = Expr(:(=), Expr(:ref, :B, :i), Expr(:call, :f, Expr(:ref, :A, :i)))
    push!(block.args, setb)
    return quote
        @turbo $loops
        return B
    end
end

@generated function _vvmapreduce!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N}
    # map_quote(N)
    map_quote()
end

################
# Version wherein an initial value is supplied

function vvmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I<:Number, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(op, Base.promote_op(f, T), Int), Dᴮ′)
    _vvmapreduce_init!(f, op, init, B, A, dims)
    return B
end

# reduction over all dims
@generated function vvmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I<:Number, T, N}
    opsym = OP.instance
    quote
        ξ = convert(Base.promote_op($opsym, Base.promote_op(f, $T), Int), init)
        @turbo for i ∈ eachindex(A)
            ξ = $opsym(f(A[i]), ξ)
        end
        return ξ
    end
end

function staticdim_mapreduce_init_quote(OP, static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if d ∈ static_dims
            push!(Bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(rinds, d)
        else
            push!(Bᵥ.args, :)
            push!(nrinds, d)
            push!(Bᵥ′.args, Symbol(:i_, d))
        end
    end
    reverse!(rinds)
    reverse!(nrinds)
    if !isempty(nrinds)
        # ξ₀ = Expr(:call, Expr(:call, :eltype, :Bᵥ), :init)
        ξ₀ = Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :init)
        block = Expr(:block)
        loops = Expr(:for, Expr(:(=), Symbol(:i_, nrinds[1]),
                                Expr(:call, :indices, Expr(:tuple, :A, :B), nrinds[1])), block)
        for i = 2:length(nrinds)
            newblock = Expr(:block)
            push!(block.args,
                  Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
                                  Expr(:call, :indices, Expr(:tuple, :A, :B), nrinds[i])), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, :ξ₀)
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :axes, :A, d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            Bᵥ = $Bᵥ
            ξ₀ = $ξ₀
            @turbo $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :init))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, Expr(:(=), Symbol(:i_, rinds[1]), Expr(:call, :axes, :A, rinds[1])), block)
        for i = 2:length(rinds)
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                                              Expr(:call, :axes, :A, rinds[i])), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        return quote
            Bᵥ = $Bᵥ
            $ξ
            @turbo $loops
            Bᵥ[] = ξ
            return B
        end
    end
end

function branches_mapreduce_init_quote(OP, N::Int, M::Int, D)
    static_dims = Int[]
    for m ∈ 1:M
        param = D.parameters[m]
        if param <: StaticInt
            new_dim = _dim(param)::Int
            push!(static_dims, new_dim)
        else
            # tuple of static dimensions
            t = Expr(:tuple)
            for n ∈ static_dims
                push!(t.args, :(StaticInt{$n}()))
            end
            q = Expr(:block, :(dimm = dims[$m]))
            qold = q
            # if-elseif statements
            ifsym = :if
            for n ∈ 1:N
                n ∈ static_dims && continue
                tc = copy(t)
                push!(tc.args, :(StaticInt{$n}()))
                qnew = Expr(ifsym, :(dimm == $n), :(return _vvmapreduce_init!(f, op, init, B, A, $tc)))
                for r ∈ m+1:M
                    push!(tc.args, :(dims[$r]))
                end
                push!(qold.args, qnew)
                qold = qnew
                ifsym = :elseif
            end
            # else statement
            tc = copy(t)
            for r ∈ m+1:M
                push!(tc.args, :(dims[$r]))
            end
            push!(qold.args, Expr(:block, :(return _vvmapreduce_init!(f, op, init, B, A, $tc))))
            return q
        end
    end
    return staticdim_mapreduce_init_quote(OP, static_dims, N)
end

@generated function _vvmapreduce_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_mapreduce_init_quote(OP, N, M, D)
end
@generated function _vvmapreduce_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N}
    # map_quote(N)
    map_quote()
end


################
# function vvmapreduce2(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
#     Dᴬ = size(A)
#     Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
#     B = similar(A, Base.promote_op(op, Base.promote_op(f, T), Int), Dᴮ′)
#     _vvmapreduce2!(f, op, init, B, A, dims)
#     return B
# end
# # reduction over all dims
# @generated function vvmapreduce2(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I, T, N}
#     fsym = F.instance
#     opsym = OP.instance
#     initsym = I.instance
#     # Tₒ = Base.promote_op(opsym, Base.promote_op(fsym, T), Int)
#     quote
#         # ξ = $initsym($Tₒ)
#         ξ = $initsym(Base.promote_op($opsym, Base.promote_op(f, $T), Int))
#         @turbo for i ∈ eachindex(A)
#             ξ = $opsym(f(A[i]), ξ)
#         end
#         return ξ
#     end
# end

# function staticdim_mapreduce2_quote(OP, I, static_dims::Vector{Int}, N::Int)
#     A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
#     Bᵥ = Expr(:call, :view, :B)
#     Bᵥ′ = Expr(:ref, :Bᵥ)
#     rinds = Int[]
#     nrinds = Int[]
#     for d = 1:N
#         if d ∈ static_dims
#             push!(Bᵥ.args, Expr(:call, :firstindex, :B, d))
#             push!(rinds, d)
#         else
#             push!(Bᵥ.args, :)
#             push!(nrinds, d)
#             push!(Bᵥ′.args, Symbol(:i_, d))
#         end
#     end
#     reverse!(rinds)
#     reverse!(nrinds)
#     if !isempty(nrinds)
#         block = Expr(:block)
#         loops = Expr(:for, Expr(:(=), Symbol(:i_, nrinds[1]),
#                                 Expr(:call, :indices, Expr(:tuple, :A, :B), nrinds[1])), block)
#         # loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B), $(nrinds[1]))), block)
#         for i = 2:length(nrinds)
#             # for d ∈ @view(nrinds[2:end])
#             newblock = Expr(:block)
#             push!(block.args,
#                   Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
#                                   Expr(:call, :indices, Expr(:tuple, :A, :B), nrinds[i])), newblock))
#             # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B), $d)), newblock))
#             block = newblock
#         end
#         rblock = block
#         # Pre-reduction
#         ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
#         push!(rblock.args, ξ)
#         # Reduction loop
#         for d ∈ rinds
#             newblock = Expr(:block)
#             push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :axes, :A, d)), newblock))
#             # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
#             block = newblock
#         end
#         # Push to inside innermost loop
#         setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance),
#                                    Expr(:call, :f, A), :ξ))
#         push!(block.args, setξ)
#         setb = Expr(:(=), Bᵥ′, :ξ)
#         push!(rblock.args, setb)
#         return quote
#             Bᵥ = $Bᵥ
#             @turbo $loops
#             return B
#         end
#     else
#         # Pre-reduction
#         ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
#         # Reduction loop
#         block = Expr(:block)
#         loops = Expr(:for, Expr(:(=), Symbol(:i_, rinds[1]),
#                                 Expr(:call, :axes, :A, rinds[1])), block)
#         # loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
#         for i = 2:length(rinds)
#             newblock = Expr(:block)
#             push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
#                                               Expr(:call, :axes, :A, rinds[i])), newblock))
#             # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
#             block = newblock
#         end
#         # Push to inside innermost loop
#         setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance),
#                                    Expr(:call, :f, A), :ξ))
#         push!(block.args, setξ)
#         return quote
#             Bᵥ = $Bᵥ
#             $ξ
#             @turbo $loops
#             Bᵥ[] = ξ
#             return B
#         end
#     end
# end

# function branches_mapreduce2_quote(OP, I, N::Int, M::Int, D)
#     static_dims = Int[]
#     for m ∈ 1:M
#         param = D.parameters[m]
#         if param <: StaticInt
#             new_dim = _dim(param)::Int
#             push!(static_dims, new_dim)
#         else
#             # tuple of static dimensions
#             t = Expr(:tuple)
#             for n ∈ static_dims
#                 push!(t.args, :(StaticInt{$n}()))
#             end
#             q = Expr(:block, :(dimm = dims[$m]))
#             qold = q
#             # if-elseif statements
#             ifsym = :if
#             for n ∈ 1:N
#                 n ∈ static_dims && continue
#                 tc = copy(t)
#                 push!(tc.args, :(StaticInt{$n}()))
#                 qnew = Expr(ifsym, :(dimm == $n), :(return _vvmapreduce2!(f, op, init, B, A, $tc)))
#                 for r ∈ m+1:M
#                     push!(tc.args, :(dims[$r]))
#                 end
#                 push!(qold.args, qnew)
#                 qold = qnew
#                 ifsym = :elseif
#             end
#             # else statement
#             tc = copy(t)
#             for r ∈ m+1:M
#                 push!(tc.args, :(dims[$r]))
#             end
#             push!(qold.args, Expr(:block, :(return _vvmapreduce2!(f, op, init, B, A, $tc))))
#             return q
#         end
#     end
#     return staticdim_mapreduce2_quote(OP, I, static_dims, N)
# end

# @generated function _vvmapreduce2!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
#     branches_mapreduce2_quote(OP, I, N, M, D)
# end
# @generated function _vvmapreduce2!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N}
#     :(copyto!(B, A); return B)
# end

############################################################################################

"""
    vtmapreduce(f, op, init, A::AbstractArray, dims=:)

Apply function `f` to each element of `A`, then reduce the result along the dimensions
`dims` using the binary function `op`. Threaded. See `vvmapreduce` for description of `dims`.
`init` need not be provided when `op` is one of `+`, `*`, `min`, `max`.

See also: [`vtsum`](@ref), [`vtprod`](@ref), [`vtminimum`](@ref), [`vtmaximum`](@ref)
"""
function vtmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(op, Base.promote_op(f, T), Int), Dᴮ′)
    _vtmapreduce!(f, op, init, B, A, dims)
    return B
end
vtmapreduce(f, op, init, A, dims::Int) = vtmapreduce(f, op, init, A, (dims,))

# dims determination would ideally be non-allocating. Also, who would
# call this anyway? Almost assuredly, a caller would already know dims, hence
# just call _vtmapreduce! anyway.
function vtmapreduce!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}) where {F, OP, I, Tₒ, T, N}
    Dᴬ = size(A)
    Dᴮ = size(B)
    dims = Tuple((d for d ∈ eachindex(Dᴮ) if isone(Dᴮ[d])))
    all(d -> Dᴮ[d] == Dᴬ[d], (d for d ∈ eachindex(Dᴮ) if !isone(Dᴮ[d]))) || throw(DimensionMismatch)
    _vtmapreduce!(f, op, init, B, A, dims)
    return B
end

# Convenience definitions
"""
    vtsum(f, A::AbstractArray, dims=:)

Sum the results of calling `f` on each element of `A` over the specified `dims`.
"""
vtsum(f::F, A, dims) where {F} = vtmapreduce(f, +, zero, A, dims)

"""
    vtprod(f, A::AbstractArray, dims=:)

Multiply the results of calling `f` on each element of `A` over the specified `dims`.
"""
vtprod(f::F, A, dims) where {F} = vtmapreduce(f, *, one, A, dims)

"""
    vtmaximum(f, A::AbstractArray, dims=:)

Compute the maximum value by calling `f` on each element of `A` over the specified `dims`.
"""
vtmaximum(f::F, A, dims) where {F} = vtmapreduce(f, max, typemin, A, dims)

"""
    vtminimum(f, A::AbstractArray, dims=:)

Compute the minimum value by calling `f` on each element of `A` over the specified `dims`.
"""
vtminimum(f::F, A, dims) where {F} = vtmapreduce(f, min, typemax, A, dims)

# ::AbstractArray required in order for kwargs interface to work
"""
    vtsum(A::AbstractArray, dims=:)

Sum the elements of `A` over the specified `dims`.
"""
vtsum(A::AbstractArray, dims) = vtmapreduce(identity, +, zero, A, dims)

"""
    vtprod(A::AbstractArray, dims=:)

Multiply the elements of `A` over the specified `dims`.
"""
vtprod(A::AbstractArray, dims) = vtmapreduce(identity, *, one, A, dims)

"""
    vtmaximum(A::AbstractArray, dims=:)

Compute the maximum value of `A` over the specified `dims`.
"""
vtmaximum(A::AbstractArray, dims) = vtmapreduce(identity, max, typemin, A, dims)

"""
    vtminimum(A::AbstractArray, dims=:)

Compute the minimum value of `A` over the specified `dims`.
"""
vtminimum(A::AbstractArray, dims) = vtmapreduce(identity, min, typemax, A, dims)


vtsum(f::F, A) where {F<:Function} = vtmapreduce(f, +, zero, A, :)
vtprod(f::F, A) where {F<:Function} = vtmapreduce(f, *, one, A, :)
vtmaximum(f::F, A) where {F<:Function} = vtmapreduce(f, max, typemin, A, :)
vtminimum(f::F, A) where {F<:Function} = vtmapreduce(f, min, typemax, A, :)

# ::AbstractArray required in order for kwargs interface to work
vtsum(A::AbstractArray) = vtmapreduce(identity, +, zero, A, :)
vtprod(A::AbstractArray) = vtmapreduce(identity, *, one, A, :)
vtmaximum(A::AbstractArray) = vtmapreduce(identity, max, typemin, A, :)
vtminimum(A::AbstractArray) = vtmapreduce(identity, min, typemax, A, :)

"""
    vtextrema(f, A::AbstractArray, dims=:)

Compute the minimum and maximum values by calling `f`  on each element of of `A`
over the specified `dims`.
"""
vtextrema(f::F, A, dims) where {F} = collect(zip(vtminimum(f, A, dims), vtmaximum(f, A, dims)))
vtextrema(f::F, A, ::Colon) where {F} = (vtminimum(f, A, :), vtmaximum(f, A, :))
vtextrema(f::F, A) where {F<:Function} = vtextrema(f, A, :)
# ::AbstractArray required in order for kwargs interface to work

"""
    vtextrema(A::AbstractArray, dims=:)

Compute the minimum and maximum values of `A` over the specified `dims`.
"""
vtextrema(A::AbstractArray, dims) = vtextrema(identity, A, dims)
vtextrema(A::AbstractArray) = (vtminimum(A), vtmaximum(A))

"""
    vtreduce(op, init, A::AbstractArray, dims=:)

Reduce `A` along the dimensions `dims` using the binary function `op`.
See `vtmapreduce` for description of `op`, `init`, `dims`.

See also: [`vtsum`](@ref), [`vtprod`](@ref), [`vtminimum`](@ref), [`vtmaximum`](@ref)
"""
vtreduce(op::OP, init::I, A, dims) where {OP, I} = vtmapreduce(identity, op, init, A, dims)
vtreduce(op::OP, init::I, A) where {OP, I} = vtmapreduce(identity, op, init, A, :)
for (op, init) ∈ zip((:+, :*, :max,:min), (:zero, :one, :typemin, :typemax))
    @eval vtreduce(::typeof($op), A; dims=:, init=$init) = vtmapreduce(identity, $op, init, A, dims)
    # 2-argument version for common binary ops, but one should just use sum,prod,maximum, minimum
    @eval vtreduce(::typeof($op), A::AbstractArray) = vtmapreduce(identity, $op, $init, A, :)
end

"""
    vtmapreduce(f, op, A; dims=:, init)

Identical to non-keyword args version; slightly less performant due to use of kwargs. Threaded.
"""
vtmapreduce(f, op, A; dims=:, init) = vtmapreduce(f, op, init, A, dims)

"""
    vtreduce(op, A; dims=:, init)

Identical to non-keyword args version; slightly less performant due to use of kwargs. Threaded.
"""
vtreduce(op, A; dims=:, init) = vtreduce(op, init, A, dims)

"""
    vtsum(f, A; dims=:, init=zero)

Sum the results of calling `f` on each element of `A` over the specified `dims`,
with the sum initialized by `init`, which may be a value `<:Number`
or a function which accepts a single type argument.
"""
vtsum(f, A; dims=:, init=zero) = vtmapreduce(f, +, init, A, dims)

"""
    vtsum(A; dims=:, init=zero)

Sum the elements of `A` over the specified `dims`, with the sum initialized by `init`.
"""
vtsum(A; dims=:, init=zero) = vtmapreduce(identity, +, init, A, dims)

"""
    vtprod(f, A; dims=:, init=one)

Multiply the results of calling `f` on each element of `A` over the specified `dims`,
with the product initialized by `init`, which may be a value `<:Number`
or a function which accepts a single type argument.
"""
vtprod(f, A; dims=:, init=one) = vtmapreduce(f, *, init, A, dims)

"""
    vtprod(A; dims=:, init=one)

Multiply the elements of `A` over the specified `dims`, with the product initialized by `init`.
"""
vtprod(A; dims=:, init=one) = vtmapreduce(identity, *, init, A, dims)

"""
    vtmaximum(f, A; dims=:, init=typemin)

Compute the maximum value of calling `f` on each element of `A` over the specified `dims`,
with the max initialized by `init`, which may be a value `<:Number`
or a function which accepts a single type argument.
"""
vtmaximum(f, A; dims=:, init=typemin) = vtmapreduce(f, max, init, A, dims)

"""
    vtmaximum(A; dims=:, init=typemin)

Compute the maximum value of `A` over the specified `dims`, with the max initialized by `init`.
"""
vtmaximum(A; dims=:, init=typemin) = vtmapreduce(identity, max, init, A, dims)

"""
    vtminimum(f, A; dims=:, init=typemax)

Compute the minimum value of calling `f` on each element of `A` over the specified `dims`,
with the min initialized by `init`, which may be a value `<:Number`
or a function which accepts a single type argument.
"""
vtminimum(f, A; dims=:, init=typemax) = vtmapreduce(f, min, init, A, dims)

"""
    vtminimum(A; dims=:, init=typemax)

Compute the minimum value of `A` over the specified `dims`, with the min initialized by `init`.
"""
vtminimum(A; dims=:, init=typemax) = vtmapreduce(identity, min, init, A, dims)

"""
    vtextrema(f, A::AbstractArray; dims=:, init=(typemax, typemin))

Compute the minimum and maximum values by calling `f`  on each element of of `A`
over the specified `dims`, with the min and max initialized by the respective arguments
of the 2-tuple `init`, which can be any combination of values `<:Number` or functions
which accept a single type argument.
"""
vtextrema(f, A; dims=:, init=(typemax, typemin)) =
    collect(zip(vtmapreduce(f, min, init[1], A, dims), vtmapreduce(f, max, init[2], A, dims)))

"""
    vtextrema(A::AbstractArray; dims=:, init=(typemax, typemin))

Compute the minimum and maximum values of `A` over the specified `dims`,
with the min and max initialized by `init`.
"""
vtextrema(A; dims=:, init=(typemax, typemin)) =
    collect(zip(vtmapreduce(identity, min, init[1], A, dims), vtmapreduce(identity, max, init[2], A, dims)))

# reduction over all dims
@generated function vtmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I, T, N}
    opsym = OP.instance
    initsym = I.instance
    quote
        ξ = $initsym(Base.promote_op($opsym, Base.promote_op(f, $T), Int))
        @tturbo for i ∈ eachindex(A)
            ξ = $opsym(f(A[i]), ξ)
        end
        return ξ
    end
end
vtmapreduce(f::F, op::OP, init::I, A) where {F, OP, I} = vtmapreduce(f, op, init, A, :)

# mixed dimensions reduction

function staticdim_tmapreduce_quote(OP, I, static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if d ∈ static_dims
            push!(Bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(rinds, d)
        else
            push!(Bᵥ.args, :)
            push!(nrinds, d)
            push!(Bᵥ′.args, Symbol(:i_, d))
        end
    end
    reverse!(rinds)
    reverse!(nrinds)
    if !isempty(nrinds)
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            Bᵥ = $Bᵥ
            @tturbo $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        return quote
            Bᵥ = $Bᵥ
            $ξ
            @tturbo $loops
            Bᵥ[] = ξ
            return B
        end
    end
end

function branches_tmapreduce_quote(OP, I, N::Int, M::Int, D)
    static_dims = Int[]
    for m ∈ 1:M
        param = D.parameters[m]
        if param <: StaticInt
            new_dim = _dim(param)::Int
            push!(static_dims, new_dim)
        else
            # tuple of static dimensions
            t = Expr(:tuple)
            for n ∈ static_dims
                push!(t.args, :(StaticInt{$n}()))
            end
            q = Expr(:block, :(dimm = dims[$m]))
            qold = q
            # if-elseif statements
            ifsym = :if
            for n ∈ 1:N
                n ∈ static_dims && continue
                tc = copy(t)
                push!(tc.args, :(StaticInt{$n}()))
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtmapreduce!(f, op, init, B, A, $tc)))
                for r ∈ m+1:M
                    push!(tc.args, :(dims[$r]))
                end
                push!(qold.args, qnew)
                qold = qnew
                ifsym = :elseif
            end
            # else statement
            tc = copy(t)
            for r ∈ m+1:M
                push!(tc.args, :(dims[$r]))
            end
            push!(qold.args, Expr(:block, :(return _vtmapreduce!(f, op, init, B, A, $tc))))
            return q
        end
    end
    return staticdim_tmapreduce_quote(OP, I, static_dims, N)
end

@generated function _vtmapreduce!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_tmapreduce_quote(OP, I, N, M, D)
end

function tmap_quote() #N::Int
    # A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    # block = Expr(:block)
    # loops = Expr(:for, Expr(:(=), Symbol(:i_, N), Expr(:call, :indices, Expr(:tuple, :A, :B), N)), block)
    # for d = N-1:-1:1
    #     newblock = Expr(:block)
    #     push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :indices, Expr(:tuple, :A, :B), d)), newblock))
    #     block = newblock
    # end
    # # Push to inside innermost loop
    # setb = Expr(:(=), Expr(:ref, :B, ntuple(d -> Symbol(:i_, d), N)...), Expr(:call, :f, A))
    # push!(block.args, setb)
    # Version using eachindex
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :A, :B)), block)
    setb = Expr(:(=), Expr(:ref, :B, :i), Expr(:call, :f, Expr(:ref, :A, :i)))
    push!(block.args, setb)
    return quote
        @tturbo $loops
        return B
    end
end

@generated function _vtmapreduce!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N}
    # tmap_quote(N)
    tmap_quote()
end

################
# Version wherein an initial value is supplied

function vtmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I<:Number, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(op, Base.promote_op(f, T), Int), Dᴮ′)
    _vtmapreduce_init!(f, op, init, B, A, dims)
    return B
end

# reduction over all dims
@generated function vtmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I<:Number, T, N}
    opsym = OP.instance
    quote
        ξ = convert(Base.promote_op($opsym, Base.promote_op(f, $T), Int), init)
        @tturbo for i ∈ eachindex(A)
            ξ = $opsym(f(A[i]), ξ)
        end
        return ξ
    end
end

function staticdim_tmapreduce_init_quote(OP, static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if d ∈ static_dims
            push!(Bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(rinds, d)
        else
            push!(Bᵥ.args, :)
            push!(nrinds, d)
            push!(Bᵥ′.args, Symbol(:i_, d))
        end
    end
    reverse!(rinds)
    reverse!(nrinds)
    if !isempty(nrinds)
        ξ₀ = Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :init)
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, :ξ₀)
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            Bᵥ = $Bᵥ
            ξ₀ = $ξ₀
            @tturbo $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :init))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        # for i = 2:length(rinds)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        return quote
            Bᵥ = $Bᵥ
            $ξ
            @tturbo $loops
            Bᵥ[] = ξ
            return B
        end
    end
end

function branches_tmapreduce_init_quote(OP, N::Int, M::Int, D)
    static_dims = Int[]
    for m ∈ 1:M
        param = D.parameters[m]
        if param <: StaticInt
            new_dim = _dim(param)::Int
            push!(static_dims, new_dim)
        else
            # tuple of static dimensions
            t = Expr(:tuple)
            for n ∈ static_dims
                push!(t.args, :(StaticInt{$n}()))
            end
            q = Expr(:block, :(dimm = dims[$m]))
            qold = q
            # if-elseif statements
            ifsym = :if
            for n ∈ 1:N
                n ∈ static_dims && continue
                tc = copy(t)
                push!(tc.args, :(StaticInt{$n}()))
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtmapreduce_init!(f, op, init, B, A, $tc)))
                for r ∈ m+1:M
                    push!(tc.args, :(dims[$r]))
                end
                push!(qold.args, qnew)
                qold = qnew
                ifsym = :elseif
            end
            # else statement
            tc = copy(t)
            for r ∈ m+1:M
                push!(tc.args, :(dims[$r]))
            end
            push!(qold.args, Expr(:block, :(return _vtmapreduce_init!(f, op, init, B, A, $tc))))
            return q
        end
    end
    return staticdim_tmapreduce_init_quote(OP, static_dims, N)
end

@generated function _vtmapreduce_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_tmapreduce_init_quote(OP, N, M, D)
end
@generated function _vtmapreduce_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N}
    # tmap_quote(N)
    tmap_quote()
end
