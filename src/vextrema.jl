#
# Date created: 2022-06-10
# Author: aradclif
#
#
############################################################################################
# interface
"""
    vvextrema([f=identity], A::AbstractArray; [dims=:], [init=(mn,mx)])

Compute the minimum and maximum values by calling `f` on each element of of `A`
over the given `dims`, with the `mn` and `mx` initialized by the respective arguments
of the 2-tuple `init`, which can be any combination of values (`<:Number`) or functions
which accept a single type argument.

# Warning
`NaN` values are not handled!

# Examples
```jldoctest
julia> A = reshape(Vector(1:2:16), (2,2,2))
2×2×2 Array{Int64, 3}:
[:, :, 1] =
 1  5
 3  7

[:, :, 2] =
  9  13
 11  15

julia> vvextrema(abs2, A, dims=(1,2), init=(typemax, 100))
1×1×2 Array{Tuple{Int64, Int64}, 3}:
[:, :, 1] =
 (1, 100)

[:, :, 2] =
 (81, 225)

julia> vvextrema(abs2, A, init=(typemax, 100))
(1, 225)
```
"""
vvextrema(f, A; dims=:, init=(typemax, typemin)) = vvextrema(f, init[1], init[2], A, dims)
vvextrema(A; dims=:, init=(typemax, typemin)) = vvextrema(identity, init[1], init[2], A, dims)

# handle convenience cases
vvextrema(f, initmin, initmax, A, dims::Int) = vvextrema(f, initmin, initmax, A, (dims,))

"""
    vvextrema([f=identity], A::AbstractArray, dims=:)

Compute the minimum and maximum values by calling `f` on each element of of `A`
over the given `dims`.
"""
vvextrema(f, A, dims) = vvextrema(f, typemax, typemin, A, dims)
vvextrema(A::AbstractArray, dims) = vvextrema(identity, A, dims)

# support any kind of init args -- this necessitates Iₘᵢₙ<:Function, Iₘₐₓ<:Function
# for the function-initialized version. It would be nice to support functors, but
# LoopVectorization would likely throw anyway.
vvextrema(f, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A, dims) where {Iₘᵢₙ<:Function, Iₘₐₓ<:Number} =
    vvextrema(f, initmin(Base.promote_op(f, eltype(A))), initmax, A, dims)
vvextrema(f, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A, dims) where {Iₘᵢₙ<:Number, Iₘₐₓ<:Function} =
    vvextrema(f, initmin, initmax(Base.promote_op(f, eltype(A))), A, dims)

vvextrema(f, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A, dims::Int) where {Iₘᵢₙ<:Function, Iₘₐₓ<:Number} =
    vvextrema(f, initmin(Base.promote_op(f, eltype(A))), initmax, A, (dims,))
vvextrema(f, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A, dims::Int) where {Iₘᵢₙ<:Number, Iₘₐₓ<:Function} =
    vvextrema(f, initmin, initmax(Base.promote_op(f, eltype(A))), A, (dims,))

################

function vvextrema(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A::AbstractArray{T, N}, ::Colon) where {F, Iₘᵢₙ<:Function, Iₘₐₓ<:Function, T, N}
    Tₒ = Base.promote_op(f, T)
    mn = initmin(Tₒ)
    mx = initmax(Tₒ)
    @turbo for i ∈ eachindex(A)
        mn = min(mn, f(A[i]))
        mx = max(mx, f(A[i]))
    end
    mn, mx
end

function vvextrema(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, Iₘᵢₙ<:Function, Iₘₐₓ<:Function, T, N, M}
    Dᴬ = size(A)
    Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    Tₒ = Base.promote_op(f, T)
    B = similar(A, Tₒ, Dᴮ)
    C = similar(A, Tₒ, Dᴮ)
    _vvextrema!(f, initmin, initmax, B, C, A, dims)
    return collect(zip(B, C))
end

function staticdim_extrema_quote(Iₘᵢₙ, Iₘₐₓ, static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    Cᵥ = Expr(:call, :view, :C)
    Cᵥ′ = Expr(:ref, :Cᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if d ∈ static_dims
            push!(Bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(Cᵥ.args, Expr(:call, :firstindex, :C, d))
            push!(rinds, d)
        else
            push!(Bᵥ.args, :)
            push!(Cᵥ.args, :)
            push!(nrinds, d)
            push!(Bᵥ′.args, Symbol(:i_, d))
            push!(Cᵥ′.args, Symbol(:i_, d))
        end
    end
    reverse!(rinds)
    reverse!(nrinds)
    if !isempty(nrinds)
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B, C), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B, C), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        mn = Expr(:(=), :mn, Expr(:call, Symbol(Iₘᵢₙ.instance), Expr(:call, :eltype, :Bᵥ)))
        mx = Expr(:(=), :mx, Expr(:call, Symbol(Iₘₐₓ.instance), Expr(:call, :eltype, :Cᵥ)))
        push!(rblock.args, mn, mx)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        setb = Expr(:(=), Bᵥ′, :mn)
        setc = Expr(:(=), Cᵥ′, :mx)
        push!(rblock.args, setb, setc)
        return quote
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            @turbo $loops
            return B, C
        end
    else
        # Pre-reduction
        mn = Expr(:(=), :mn, Expr(:call, Symbol(Iₘᵢₙ.instance), Expr(:call, :eltype, :Bᵥ)))
        mx = Expr(:(=), :mx, Expr(:call, Symbol(Iₘₐₓ.instance), Expr(:call, :eltype, :Cᵥ)))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        return quote
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            $mn
            $mx
            @turbo $loops
            Bᵥ[] = mn
            Cᵥ[] = mx
            return B, C
        end
    end
end

function branches_extrema_quote(Iₘᵢₙ, Iₘₐₓ, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vvextrema!(f, initmin, initmax, B, C, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vvextrema!(f, initmin, initmax, B, C, A, $tc))))
            return q
        end
    end
    return staticdim_extrema_quote(Iₘᵢₙ, Iₘₐₓ, static_dims, N)
end

@generated function _vvextrema!(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, Iₘᵢₙ, Iₘₐₓ, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_extrema_quote(Iₘᵢₙ, Iₘₐₓ, N, M, D)
end

function extrema_map_quote(Iₘᵢₙ, Iₘₐₓ)
    # Pre-reduction
    mn = Expr(:(=), :mn, Expr(:call, Symbol(Iₘᵢₙ.instance), Expr(:call, :eltype, :B)))
    mx = Expr(:(=), :mx, Expr(:call, Symbol(Iₘₐₓ.instance), Expr(:call, :eltype, :C)))
    # Reduction loop
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :A, :B, :C)), block)
    setb = Expr(:(=), Expr(:ref, :B, :i), Expr(:call, :min, :mn, Expr(:call, :f, Expr(:ref, :A, :i))))
    setc = Expr(:(=), Expr(:ref, :C, :i), Expr(:call, :max, :mx, Expr(:call, :f, Expr(:ref, :A, :i))))
    push!(block.args, setb, setc)
    return quote
        $mn
        $mx
        @turbo $loops
        return B, C
    end
end

@generated function _vvextrema!(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, Iₘᵢₙ, Iₘₐₓ, Tₒ, T, N, M}
    extrema_map_quote(Iₘᵢₙ, Iₘₐₓ)
end

################
# Version wherein an initial value is supplied
function vvextrema(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A::AbstractArray{T, N}, ::Colon) where {F, Iₘᵢₙ<:Number, Iₘₐₓ<:Number, T, N}
    Tₒ = Base.promote_op(f, T)
    mn = convert(Tₒ, initmin)
    mx = convert(Tₒ, initmax)
    @turbo for i ∈ eachindex(A)
        mn = min(mn, f(A[i]))
        mx = max(mx, f(A[i]))
    end
    mn, mx
end

function vvextrema(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, Iₘᵢₙ<:Number, Iₘₐₓ<:Number, T, N, M}
    Dᴬ = size(A)
    Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    Tₒ = Base.promote_op(f, T)
    B = similar(A, Tₒ, Dᴮ)
    C = similar(A, Tₒ, Dᴮ)
    _vvextrema_init!(f, initmin, initmax, B, C, A, dims)
    return collect(zip(B, C))
end

function staticdim_extrema_init_quote(static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    Cᵥ = Expr(:call, :view, :C)
    Cᵥ′ = Expr(:ref, :Cᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if d ∈ static_dims
            push!(Bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(Cᵥ.args, Expr(:call, :firstindex, :C, d))
            push!(rinds, d)
        else
            push!(Bᵥ.args, :)
            push!(Cᵥ.args, :)
            push!(nrinds, d)
            push!(Bᵥ′.args, Symbol(:i_, d))
            push!(Cᵥ′.args, Symbol(:i_, d))
        end
    end
    reverse!(rinds)
    reverse!(nrinds)
    if !isempty(nrinds)
        mn₀ = Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :initmin)
        mx₀ = Expr(:call, :convert, Expr(:call, :eltype, :Cᵥ), :initmax)
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B, C), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B, C), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        mn = Expr(:(=), :mn, :mn₀)
        mx = Expr(:(=), :mx, :mx₀)
        push!(rblock.args, mn, mx)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        setb = Expr(:(=), Bᵥ′, :mn)
        setc = Expr(:(=), Cᵥ′, :mx)
        push!(rblock.args, setb, setc)
        return quote
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            mn₀ = $mn₀
            mx₀ = $mx₀
            @turbo $loops
            return B, C
        end
    else
        # Pre-reduction
        mn₀ = Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :initmin)
        mx₀ = Expr(:call, :convert, Expr(:call, :eltype, :Cᵥ), :initmax)
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        return quote
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            mn = $mn₀
            mx = $mx₀
            @turbo $loops
            Bᵥ[] = mn
            Cᵥ[] = mx
            return B, C
        end
    end
end

function branches_extrema_init_quote(N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vvextrema_init!(f, initmin, initmax, B, C, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vvextrema_init!(f, initmin, initmax, B, C, A, $tc))))
            return q
        end
    end
    return staticdim_extrema_init_quote(static_dims, N)
end


@generated function _vvextrema_init!(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, Iₘᵢₙ, Iₘₐₓ, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_extrema_init_quote(N, M, D)
end

function extrema_init_map_quote()
    # Pre-reduction
    mn₀ = Expr(:call, :convert, Expr(:call, :eltype, :B), :initmin)
    mx₀ = Expr(:call, :convert, Expr(:call, :eltype, :C), :initmax)
    # Reduction loop
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :A, :B, :C)), block)
    setb = Expr(:(=), Expr(:ref, :B, :i), Expr(:call, :min, :mn, Expr(:call, :f, Expr(:ref, :A, :i))))
    setc = Expr(:(=), Expr(:ref, :C, :i), Expr(:call, :max, :mx, Expr(:call, :f, Expr(:ref, :A, :i))))
    push!(block.args, setb, setc)
    return quote
        mn = $mn₀
        mx = $mx₀
        @turbo $loops
        return B, C
    end
end

@generated function _vvextrema_init!(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, Iₘᵢₙ, Iₘₐₓ, Tₒ, T, N, M}
    extrema_init_map_quote()
end

############################################################################################
"""
    vtextrema([f=identity], A::AbstractArray; [dims=:], [init=(mn,mx)])

Compute the minimum and maximum values by calling `f` on each element of of `A`
over the given `dims`, with the `mn` and `mn` initialized by the respective arguments
of the 2-tuple `init`, which can be any combination of values (`<:Number`) or functions
which accept a single type argument. Threaded.

# Warning
`NaN` values are not handled!

# Examples
```jldoctest
julia> A = reshape(Vector(1:2:16), (2,2,2))
2×2×2 Array{Int64, 3}:
[:, :, 1] =
 1  5
 3  7

[:, :, 2] =
  9  13
 11  15

julia> vtextrema(abs2, A, dims=(1,2), init=(typemax, 100))
1×1×2 Array{Tuple{Int64, Int64}, 3}:
[:, :, 1] =
 (1, 100)

[:, :, 2] =
 (81, 225)

julia> vtextrema(abs2, A, init=(typemax, 100))
(1, 225)
```
"""
vtextrema(f, A; dims=:, init=(typemax, typemin)) = vtextrema(f, init[1], init[2], A, dims)
vtextrema(A; dims=:, init=(typemax, typemin)) = vtextrema(identity, init[1], init[2], A, dims)

# handle convenience cases
vtextrema(f, initmin, initmax, A, dims::Int) = vtextrema(f, initmin, initmax, A, (dims,))

"""
    vtextrema([f=identity], A::AbstractArray, dims=:)

Compute the minimum and maximum values by calling `f` on each element of of `A`
over the given `dims`. Threaded.
"""
vtextrema(f, A, dims) = vtextrema(f, typemax, typemin, A, dims)
vtextrema(A::AbstractArray, dims) = vtextrema(identity, A, dims)

# support any kind of init args -- this necessitates Iₘᵢₙ<:Function, Iₘₐₓ<:Function
# for the function-initialized version. It would be nice to support functors, but
# LoopVectorization would likely throw anyway.
vtextrema(f, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A, dims) where {Iₘᵢₙ<:Function, Iₘₐₓ<:Number} =
    vtextrema(f, initmin(Base.promote_op(f, eltype(A))), initmax, A, dims)
vtextrema(f, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A, dims) where {Iₘᵢₙ<:Number, Iₘₐₓ<:Function} =
    vtextrema(f, initmin, initmax(Base.promote_op(f, eltype(A))), A, dims)

vtextrema(f, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A, dims::Int) where {Iₘᵢₙ<:Function, Iₘₐₓ<:Number} =
    vtextrema(f, initmin(Base.promote_op(f, eltype(A))), initmax, A, (dims,))
vtextrema(f, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A, dims::Int) where {Iₘᵢₙ<:Number, Iₘₐₓ<:Function} =
    vtextrema(f, initmin, initmax(Base.promote_op(f, eltype(A))), A, (dims,))
################

function vtextrema(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A::AbstractArray{T, N}, ::Colon) where {F, Iₘᵢₙ<:Function, Iₘₐₓ<:Function, T, N}
    Tₒ = Base.promote_op(f, T)
    mn = initmin(Tₒ)
    mx = initmax(Tₒ)
    @tturbo for i ∈ eachindex(A)
        mn = min(mn, f(A[i]))
        mx = max(mx, f(A[i]))
    end
    mn, mx
end

function vtextrema(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, Iₘᵢₙ<:Function, Iₘₐₓ<:Function, T, N, M}
    Dᴬ = size(A)
    Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    Tₒ = Base.promote_op(f, T)
    B = similar(A, Tₒ, Dᴮ)
    C = similar(A, Tₒ, Dᴮ)
    _vtextrema!(f, initmin, initmax, B, C, A, dims)
    return collect(zip(B, C))
end

function staticdim_textrema_quote(Iₘᵢₙ, Iₘₐₓ, static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    Cᵥ = Expr(:call, :view, :C)
    Cᵥ′ = Expr(:ref, :Cᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if d ∈ static_dims
            push!(Bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(Cᵥ.args, Expr(:call, :firstindex, :C, d))
            push!(rinds, d)
        else
            push!(Bᵥ.args, :)
            push!(Cᵥ.args, :)
            push!(nrinds, d)
            push!(Bᵥ′.args, Symbol(:i_, d))
            push!(Cᵥ′.args, Symbol(:i_, d))
        end
    end
    reverse!(rinds)
    reverse!(nrinds)
    if !isempty(nrinds)
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B, C), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B, C), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        mn = Expr(:(=), :mn, Expr(:call, Symbol(Iₘᵢₙ.instance), Expr(:call, :eltype, :Bᵥ)))
        mx = Expr(:(=), :mx, Expr(:call, Symbol(Iₘₐₓ.instance), Expr(:call, :eltype, :Cᵥ)))
        push!(rblock.args, mn, mx)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        setb = Expr(:(=), Bᵥ′, :mn)
        setc = Expr(:(=), Cᵥ′, :mx)
        push!(rblock.args, setb, setc)
        return quote
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            @tturbo $loops
            return B, C
        end
    else
        # Pre-reduction
        mn = Expr(:(=), :mn, Expr(:call, Symbol(Iₘᵢₙ.instance), Expr(:call, :eltype, :Bᵥ)))
        mx = Expr(:(=), :mx, Expr(:call, Symbol(Iₘₐₓ.instance), Expr(:call, :eltype, :Cᵥ)))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        return quote
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            $mn
            $mx
            @tturbo $loops
            Bᵥ[] = mn
            Cᵥ[] = mx
            return B, C
        end
    end
end

function branches_textrema_quote(Iₘᵢₙ, Iₘₐₓ, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtextrema!(f, initmin, initmax, B, C, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vtextrema!(f, initmin, initmax, B, C, A, $tc))))
            return q
        end
    end
    return staticdim_textrema_quote(Iₘᵢₙ, Iₘₐₓ, static_dims, N)
end

@generated function _vtextrema!(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, Iₘᵢₙ, Iₘₐₓ, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_textrema_quote(Iₘᵢₙ, Iₘₐₓ, N, M, D)
end

function textrema_map_quote(Iₘᵢₙ, Iₘₐₓ)
    # Pre-reduction
    mn = Expr(:(=), :mn, Expr(:call, Symbol(Iₘᵢₙ.instance), Expr(:call, :eltype, :B)))
    mx = Expr(:(=), :mx, Expr(:call, Symbol(Iₘₐₓ.instance), Expr(:call, :eltype, :C)))
    # Reduction loop
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :A, :B, :C)), block)
    setb = Expr(:(=), Expr(:ref, :B, :i), Expr(:call, :min, :mn, Expr(:call, :f, Expr(:ref, :A, :i))))
    setc = Expr(:(=), Expr(:ref, :C, :i), Expr(:call, :max, :mx, Expr(:call, :f, Expr(:ref, :A, :i))))
    push!(block.args, setb, setc)
    return quote
        $mn
        $mx
        @tturbo $loops
        return B, C
    end
end

@generated function _vtextrema!(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, Iₘᵢₙ, Iₘₐₓ, Tₒ, T, N, M}
    textrema_map_quote(Iₘᵢₙ, Iₘₐₓ)
end

############################################################################################
# Version wherein an initial value is supplied
function vtextrema(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A::AbstractArray{T, N}, ::Colon) where {F, Iₘᵢₙ<:Number, Iₘₐₓ<:Number, T, N}
    Tₒ = Base.promote_op(f, T)
    mn = convert(Tₒ, initmin)
    mx = convert(Tₒ, initmax)
    @tturbo for i ∈ eachindex(A)
        mn = min(mn, f(A[i]))
        mx = max(mx, f(A[i]))
    end
    mn, mx
end

function vtextrema(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, Iₘᵢₙ<:Number, Iₘₐₓ<:Number, T, N, M}
    Dᴬ = size(A)
    Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    Tₒ = Base.promote_op(f, T)
    B = similar(A, Tₒ, Dᴮ)
    C = similar(A, Tₒ, Dᴮ)
    _vtextrema_init!(f, initmin, initmax, B, C, A, dims)
    return collect(zip(B, C))
end

function staticdim_textrema_init_quote(static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    Cᵥ = Expr(:call, :view, :C)
    Cᵥ′ = Expr(:ref, :Cᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if d ∈ static_dims
            push!(Bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(Cᵥ.args, Expr(:call, :firstindex, :C, d))
            push!(rinds, d)
        else
            push!(Bᵥ.args, :)
            push!(Cᵥ.args, :)
            push!(nrinds, d)
            push!(Bᵥ′.args, Symbol(:i_, d))
            push!(Cᵥ′.args, Symbol(:i_, d))
        end
    end
    reverse!(rinds)
    reverse!(nrinds)
    if !isempty(nrinds)
        mn₀ = Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :initmin)
        mx₀ = Expr(:call, :convert, Expr(:call, :eltype, :Cᵥ), :initmax)
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B, C), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B, C), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        mn = Expr(:(=), :mn, :mn₀)
        mx = Expr(:(=), :mx, :mx₀)
        push!(rblock.args, mn, mx)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        setb = Expr(:(=), Bᵥ′, :mn)
        setc = Expr(:(=), Cᵥ′, :mx)
        push!(rblock.args, setb, setc)
        return quote
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            mn₀ = $mn₀
            mx₀ = $mx₀
            @tturbo $loops
            return B, C
        end
    else
        # Pre-reduction
        mn₀ = Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :initmin)
        mx₀ = Expr(:call, :convert, Expr(:call, :eltype, :Cᵥ), :initmax)
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        return quote
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            mn = $mn₀
            mx = $mx₀
            @tturbo $loops
            Bᵥ[] = mn
            Cᵥ[] = mx
            return B, C
        end
    end
end

function branches_textrema_init_quote(N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtextrema_init!(f, initmin, initmax, B, C, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vtextrema_init!(f, initmin, initmax, B, C, A, $tc))))
            return q
        end
    end
    return staticdim_textrema_init_quote(static_dims, N)
end


@generated function _vtextrema_init!(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, Iₘᵢₙ, Iₘₐₓ, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_textrema_init_quote(N, M, D)
end

function textrema_init_map_quote()
    # Pre-reduction
    mn₀ = Expr(:call, :convert, Expr(:call, :eltype, :B), :initmin)
    mx₀ = Expr(:call, :convert, Expr(:call, :eltype, :C), :initmax)
    # Reduction loop
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :A, :B, :C)), block)
    setb = Expr(:(=), Expr(:ref, :B, :i), Expr(:call, :min, :mn, Expr(:call, :f, Expr(:ref, :A, :i))))
    setc = Expr(:(=), Expr(:ref, :C, :i), Expr(:call, :max, :mx, Expr(:call, :f, Expr(:ref, :A, :i))))
    push!(block.args, setb, setc)
    return quote
        mn = $mn₀
        mx = $mx₀
        @tturbo $loops
        return B, C
    end
end

@generated function _vtextrema_init!(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, Iₘᵢₙ, Iₘₐₓ, Tₒ, T, N, M}
    textrema_init_map_quote()
end
