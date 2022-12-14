#
# Date created: 2022-09-13
# Author: aradclif
#
#
############################################################################################

"""
    vmapreducethen(f, op, g, As::Vararg{AbstractArray, N}; dims=:, init) where {N}

Version of `mapreducethen` wherein `f` : ℝᴺ → ℝ, then `g` : ℝ → ℝ, with reduction over
the dimensions `dims`.

# Examples
```jldoctest
julia> x, y, z = [1 2; 3 4], [5 6; 7 8], [9 10; 11 12];

julia> vmapreducethen((a, b) -> abs2(a - b), +, √, x, y, dims=2)    # Euclidean distance
2×1 Matrix{Float64}:
 5.656854249492381
 5.656854249492381

julia> vmapreducethen(*, *, exp, x, y, z, dims=2, init=-1.0)
2×1 Matrix{Float64}:
 0.0
 0.0
```
"""
vmapreducethen(f::F, op::OP, g::G, init::I, As::Vararg{AbstractArray, P}) where {F, OP, G, I, P} =
    vmapreducethen(f, op, init, As, :)

vmapreducethen(f, op, g, As::Vararg{AbstractArray, P}; dims=:, init) where {P} =
    vmapreducethen(f, op, g, init, As, dims)

function vmapreducethen(f::F, op::OP, g::G, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, G, I, M, P}
    ax = axes(As[1])
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), ndims(As[1]))
    B = similar(As[1], Base.promote_op(g, Base.promote_op(op, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int)), Dᴮ′)
    _vmapreducethen_vararg!(f, op, g, init, B, As, dims)
    return B
end

function staticdim_mapreducethen_vararg_quote(OP, I, static_dims::Vector{Int}, N::Int, P::Int)
    t = Expr(:tuple)
    for p = 1:P
        push!(t.args, Symbol(:A_, p))
    end
    f = Expr(:call, :f)
    for p = 1:P
        A = Expr(:ref, Symbol(:A_, p), ntuple(d -> Symbol(:i_, d), N)...)
        push!(f.args, A)
    end
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
        tc = copy(t)
        push!(tc.args, :B)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices($tc, $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($tc, $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        # ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :𝒯ₒ))
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, Expr(:call, :g, :ξ))
        push!(rblock.args, setb)
        return quote
            $t = As
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int)
            Bᵥ = $Bᵥ
            @turbo $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :𝒯ₒ))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = indices($t, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        return quote
            $t = As
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int)
            Bᵥ = $Bᵥ
            $ξ
            @turbo $loops
            Bᵥ[] = g(ξ)
            return B
        end
    end
end

function branches_mapreducethen_vararg_quote(OP, I, N::Int, M::Int, P::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vmapreducethen_vararg!(f, op, g, init, B, As, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vmapreducethen_vararg!(f, op, g, init, B, As, $tc))))
            return q
        end
    end
    return staticdim_mapreducethen_vararg_quote(OP, I, static_dims, N, P)
end

# In the case of rinds = ∅, this just corresponds to a map
function mapthen_vararg_quote(P::Int)
    t = Expr(:tuple)
    f = Expr(:call, :f)
    ei = Expr(:call, :eachindex)
    for p = 1:P
        Asym = Symbol(:A_, p)
        A = Expr(:ref, Asym, :i)
        push!(t.args, Asym)
        push!(f.args, A)
        push!(ei.args, Asym)
    end
    push!(ei.args, :B)
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, ei), block)
    setb = Expr(:(=), Expr(:ref, :B, :i), Expr(:call, :g, f))
    push!(block.args, setb)
    return quote
        $t = As
        @turbo $loops
        return B
    end
end

@generated function _vmapreducethen_vararg!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, G, I, Tₒ, N, M, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_mapreducethen_vararg_quote(OP, I, N, M, P, D)
end

@generated function _vmapreducethen_vararg!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, G, I, Tₒ, N, P}
    mapthen_vararg_quote(P)
end

function vmapreducethen(f::F, op::OP, g::G, init::I, As::Tuple{Vararg{AbstractArray, P}}, ::Colon) where {F, G, OP, I, P}
    _mapreducethenall_vararg(f, op, g, init, As)
end

function mapreducethenall_vararg_quote(OP, I, P::Int)
    t = Expr(:tuple)
    for p = 1:P
        push!(t.args, Symbol(:A_, p))
    end
    f = Expr(:call, :f)
    for p = 1:P
        A = Expr(:ref, Symbol(:A_, p), :i)
        push!(f.args, A)
    end
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, t.args...)), block)
    setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
    push!(block.args, setξ)
    # Pre-reduction
    ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :(Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int))))
    return quote
        $t = As
        $ξ
        @turbo $loops
        return g(ξ)
    end
end

@generated function _mapreducethenall_vararg(f::F, op::OP, g::G, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, G, I, P}
    mapreducethenall_vararg_quote(OP, I, P)
end

################
# Version wherein an initial value is supplied

function vmapreducethen(f::F, op::OP, g::G, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, G, I<:Number, M, P}
    ax = axes(As[1])
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), ndims(As[1]))
    B = similar(As[1], Base.promote_op(g, Base.promote_op(op, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int)), Dᴮ′)
    _vmapreducethen_vararg_init!(f, op, g, init, B, As, dims)
end

function staticdim_mapreducethen_vararg_init_quote(OP, static_dims::Vector{Int}, N::Int, P::Int)
    t = Expr(:tuple)
    for p = 1:P
        push!(t.args, Symbol(:A_, p))
    end
    f = Expr(:call, :f)
    for p = 1:P
        A = Expr(:ref, Symbol(:A_, p), ntuple(d -> Symbol(:i_, d), N)...)
        push!(f.args, A)
    end
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
        ξ₀ = Expr(:call, :convert, :𝒯ₒ, :init)
        block = Expr(:block)
        tc = copy(t)
        push!(tc.args, :B)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices($tc, $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($tc, $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, :ξ₀)
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, Expr(:call, :g, :ξ))
        push!(rblock.args, setb)
        return quote
            $t = As
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int)
            Bᵥ = $Bᵥ
            ξ₀ = $ξ₀
            @turbo $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, :convert, :𝒯ₒ, :init))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = indices($t, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        return quote
            $t = As
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int)
            Bᵥ = $Bᵥ
            $ξ
            @turbo $loops
            Bᵥ[] = g(ξ)
            return B
        end
    end
end

function branches_mapreducethen_vararg_init_quote(OP, N::Int, M::Int, P::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vmapreducethen_vararg_init!(f, op, g, init, B, As, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vmapreducethen_vararg_init!(f, op, g, init, B, As, $tc))))
            return q
        end
    end
    return staticdim_mapreducethen_vararg_init_quote(OP, static_dims, N, P)
end

@generated function _vmapreducethen_vararg_init!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, G, I, Tₒ, N, M, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_mapreducethen_vararg_init_quote(OP, N, M, P, D)
end

@generated function _vmapreducethen_vararg_init!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, G, I, Tₒ, N, P}
    mapthen_vararg_quote(P)
end

function mapreducethenall_vararg_init_quote(OP, P::Int)
    t = Expr(:tuple)
    for p = 1:P
        push!(t.args, Symbol(:A_, p))
    end
    f = Expr(:call, :f)
    for p = 1:P
        A = Expr(:ref, Symbol(:A_, p), :i)
        push!(f.args, A)
    end
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, t.args...)), block)
    setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
    push!(block.args, setξ)
    # Pre-reduction
    ξ = Expr(:(=), :ξ, Expr(:call, :convert, :(Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int)), :init))
    return quote
        $t = As
        $ξ
        @turbo $loops
        return g(ξ)
    end
end

@generated function _mapreducethenall_vararg(f::F, op::OP, g::G, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, G, I<:Number, P}
    mapreducethenall_vararg_init_quote(OP, P)
end

################
# Threaded version

"""
    vtmapreducethen(f, op, g, init, As::Vararg{AbstractArray, N}) where {N}

Version of `mapreducethen` for `f` : ℝᴺ → ℝ, then `g` : ℝ → ℝ, with reduction occurring over
all dimensions.
"""
vtmapreducethen(f::F, op::OP, g::G, init::I, As::Vararg{AbstractArray, P}) where {F, OP, G, I, P} =
    vtmapreducethen(f, op, init, As, :)

"""
    vtmapreducethen(f, op, g, As::Vararg{AbstractArray, N}; dims=:, init) where {N}

Keyword args version for `f` : ℝᴺ → ℝ, then `g` : ℝ → ℝ.
"""
vtmapreducethen(f, op, g, As::Vararg{AbstractArray, P}; dims=:, init) where {P} =
    vtmapreducethen(f, op, g, init, As, dims)

"""
    vtmapreducethen(f, op, g, init, As::Tuple{Vararg{AbstractArray}}, dims=:)

Version of `mapreducethen` for `f` : ℝᴺ → ℝ, then `g` : ℝ → ℝ, with reduction over given `dims`.
"""
function vtmapreducethen(f::F, op::OP, g::G, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, G, I, M, P}
    ax = axes(As[1])
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), ndims(As[1]))
    B = similar(As[1], Base.promote_op(g, Base.promote_op(op, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int)), Dᴮ′)
    _vtmapreducethen_vararg!(f, op, g, init, B, As, dims)
    return B
end

function staticdim_tmapreducethen_vararg_quote(OP, I, static_dims::Vector{Int}, N::Int, P::Int)
    t = Expr(:tuple)
    for p = 1:P
        push!(t.args, Symbol(:A_, p))
    end
    f = Expr(:call, :f)
    for p = 1:P
        A = Expr(:ref, Symbol(:A_, p), ntuple(d -> Symbol(:i_, d), N)...)
        push!(f.args, A)
    end
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
        tc = copy(t)
        push!(tc.args, :B)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices($tc, $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($tc, $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        # ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :𝒯ₒ))
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, Expr(:call, :g, :ξ))
        push!(rblock.args, setb)
        return quote
            $t = As
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int)
            Bᵥ = $Bᵥ
            @tturbo $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :𝒯ₒ))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = indices($t, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        return quote
            $t = As
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int)
            Bᵥ = $Bᵥ
            $ξ
            @tturbo $loops
            Bᵥ[] = g(ξ)
            return B
        end
    end
end

function branches_tmapreducethen_vararg_quote(OP, I, N::Int, M::Int, P::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtmapreducethen_vararg!(f, op, g, init, B, As, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vtmapreducethen_vararg!(f, op, g, init, B, As, $tc))))
            return q
        end
    end
    return staticdim_tmapreducethen_vararg_quote(OP, I, static_dims, N, P)
end

# In the case of rinds = ∅, this just corresponds to a map
function tmapthen_vararg_quote(P::Int)
    t = Expr(:tuple)
    f = Expr(:call, :f)
    ei = Expr(:call, :eachindex)
    for p = 1:P
        Asym = Symbol(:A_, p)
        A = Expr(:ref, Asym, :i)
        push!(t.args, Asym)
        push!(f.args, A)
        push!(ei.args, Asym)
    end
    push!(ei.args, :B)
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, ei), block)
    setb = Expr(:(=), Expr(:ref, :B, :i), Expr(:call, :g, f))
    push!(block.args, setb)
    return quote
        $t = As
        @tturbo $loops
        return B
    end
end

@generated function _vtmapreducethen_vararg!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, G, I, Tₒ, N, M, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_tmapreducethen_vararg_quote(OP, I, N, M, P, D)
end

@generated function _vtmapreducethen_vararg!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, G, I, Tₒ, N, P}
    tmapthen_vararg_quote(P)
end

function vtmapreducethen(f::F, op::OP, g::G, init::I, As::Tuple{Vararg{AbstractArray, P}}, ::Colon) where {F, G, OP, I, P}
    _tmapreducethenall_vararg(f, op, g, init, As)
end

function tmapreducethenall_vararg_quote(OP, I, P::Int)
    t = Expr(:tuple)
    for p = 1:P
        push!(t.args, Symbol(:A_, p))
    end
    f = Expr(:call, :f)
    for p = 1:P
        A = Expr(:ref, Symbol(:A_, p), :i)
        push!(f.args, A)
    end
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, t.args...)), block)
    setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
    push!(block.args, setξ)
    # Pre-reduction
    ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :(Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int))))
    return quote
        $t = As
        $ξ
        @tturbo $loops
        return g(ξ)
    end
end

@generated function _tmapreducethenall_vararg(f::F, op::OP, g::G, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, G, I, P}
    tmapreducethenall_vararg_quote(OP, I, P)
end

################
# Version wherein an initial value is supplied

function vtmapreducethen(f::F, op::OP, g::G, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, G, I<:Number, M, P}
    ax = axes(As[1])
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), ndims(As[1]))
    B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int), Dᴮ′)
    _vtmapreducethen_vararg_init!(f, op, g, init, B, As, dims)
end

function staticdim_tmapreducethen_vararg_init_quote(OP, static_dims::Vector{Int}, N::Int, P::Int)
    t = Expr(:tuple)
    for p = 1:P
        push!(t.args, Symbol(:A_, p))
    end
    f = Expr(:call, :f)
    for p = 1:P
        A = Expr(:ref, Symbol(:A_, p), ntuple(d -> Symbol(:i_, d), N)...)
        push!(f.args, A)
    end
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
        ξ₀ = Expr(:call, :convert, :𝒯ₒ, :init)
        block = Expr(:block)
        tc = copy(t)
        push!(tc.args, :B)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices($tc, $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($tc, $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, :ξ₀)
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, Expr(:call, :g, :ξ))
        push!(rblock.args, setb)
        return quote
            $t = As
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int)
            Bᵥ = $Bᵥ
            ξ₀ = $ξ₀
            @tturbo $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, :convert, :𝒯ₒ, :init))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = indices($t, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        return quote
            $t = As
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int)
            Bᵥ = $Bᵥ
            $ξ
            @tturbo $loops
            Bᵥ[] = g(ξ)
            return B
        end
    end
end

function branches_tmapreducethen_vararg_init_quote(OP, N::Int, M::Int, P::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtmapreducethen_vararg_init!(f, op, g, init, B, As, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vtmapreducethen_vararg_init!(f, op, g, init, B, As, $tc))))
            return q
        end
    end
    return staticdim_tmapreducethen_vararg_init_quote(OP, static_dims, N, P)
end

@generated function _vtmapreducethen_vararg_init!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, G, I, Tₒ, N, M, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_tmapreducethen_vararg_init_quote(OP, N, M, P, D)
end

@generated function _vtmapreducethen_vararg_init!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, G, I, Tₒ, N, P}
    tmapthen_vararg_quote(P)
end

function tmapreducethenall_vararg_init_quote(OP, P::Int)
    t = Expr(:tuple)
    for p = 1:P
        push!(t.args, Symbol(:A_, p))
    end
    f = Expr(:call, :f)
    for p = 1:P
        A = Expr(:ref, Symbol(:A_, p), :i)
        push!(f.args, A)
    end
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, t.args...)), block)
    setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
    push!(block.args, setξ)
    # Pre-reduction
    ξ = Expr(:(=), :ξ, Expr(:call, :convert, :(Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int)), :init))
    return quote
        $t = As
        $ξ
        @tturbo $loops
        return g(ξ)
    end
end

@generated function _tmapreducethenall_vararg(f::F, op::OP, g::G, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, G, I<:Number, P}
    tmapreducethenall_vararg_init_quote(OP, P)
end
