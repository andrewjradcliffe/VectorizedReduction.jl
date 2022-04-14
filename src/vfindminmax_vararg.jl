#
# Date created: 2022-03-24
# Author: aradclif
#
#
############################################################################################
# Version of findmin, findmax for `f`: ℝᴺ -> ℝ
# Note: it is only defined with for such functions.

# Necessary dispatches to define interface
vfindmax(f::F, As::Vararg{AbstractArray, P}) where {F<:Function, P} =
    vfindminmax(f, >, typemin, As, :)
vfindmin(f::F, As::Vararg{AbstractArray, P}) where {F<:Function, P} =
    vfindminmax(f, <, typemax, As, :)

vfindmax(f, As::Vararg{AbstractArray, P}; dims=:) where {P} = vfindminmax(f, >, typemin, As, dims)
vfindmin(f, As::Vararg{AbstractArray, P}; dims=:) where {P} = vfindminmax(f, <, typemax, As, dims)

function vfindminmax(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, I, M, P}
    A₁ = As[1]
    ax = axes(A₁)
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴬ = size(A₁)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], ndims(A₁))
    B = similar(A₁, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Dᴮ′)
    C = similar(A₁, Int, Dᴮ′)
    _vfindminmax_vararg!(f, op, init, B, C, As, dims)
    return B, CartesianIndices(A₁)[C]
end

# Not necessary
# vfindmax(f::F, As::Tuple{Vararg{AbstractArray, P}}, dims) where {F, P} =
#     vfindminmax(f, >, typemin, As, dims)
# vfindmin(f::F, As::Tuple{Vararg{AbstractArray, P}}, dims) where {F, P} =
#     vfindminmax(f, <, typemax, As, dims)

# Over all dims, i.e. if dims = :
function vfindminmax(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, ::Colon) where {F, OP, I, P}
    _findminmaxall_vararg(f, op, init, As)
end
function vfindminmax(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractVector, P}}, ::Colon) where {F, OP, I, P}
    ξ, CI = _findminmaxall_vararg(f, op, init, As)
    return ξ, CI[1] # LinearIndices(As[1])[CI]
end

function findminmaxall_vararg_quote(OP, I, P::Int)
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
    cmpr = Expr(:(=), :newmax, Expr(:call, Symbol(OP.instance), f, :ξ))
    push!(block.args, cmpr)
    setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax, f, :ξ))
    push!(block.args, setmax)
    setindmax = Expr(:(=), :indmax, Expr(:call, :ifelse, :newmax, :i, :indmax))
    push!(block.args, setindmax)
    # Pre-reduction
    ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance),
                            :(Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)))))
    return quote
        $t = As
        indmax = 1
        $ξ
        @turbo $loops
        return ξ, CartesianIndices(A_1)[indmax]
    end
end

@generated function _findminmaxall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, I, P}
    findminmaxall_vararg_quote(OP, I, P)
end
# Not necessary either
# vfindmax(f::F, As::Tuple{Vararg{AbstractArray, P}}) where {F<:Function, P} =
#     vfindminmax(f, >, typemin, As, :)
# vfindmin(f::F, As::Tuple{Vararg{AbstractArray, P}}) where {F<:Function, P} =
#     vfindminmax(f, <, typemax, As, :)

function staticdim_findminmax_vararg_quote(OP, I, static_dims::Vector{Int}, N::Int, P::Int)
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
    Cᵥ = Expr(:call, :view, :C)
    Bᵥ′ = Expr(:ref, :Bᵥ)
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
        tc = copy(t)
        push!(tc.args, :B)
        push!(tc.args, :C)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices($tc, $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($tc, $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        push!(rblock.args, ξ)
        for d ∈ rinds
            push!(rblock.args, Expr(:(=), Symbol(:j_, d), Expr(:call, :one, :Int)))
        end
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        cmpr = Expr(:(=), :newmax, Expr(:call, Symbol(OP.instance), f, :ξ))
        push!(block.args, cmpr)
        setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax, f, :ξ))
        push!(block.args, setmax)
        for d ∈ rinds
            setj = Expr(:(=), Symbol(:j_, d),
                        Expr(:call, :ifelse, :newmax, Symbol(:i_, d), Symbol(:j_, d)))
            # setj = :($(Symbol(:j_, d)) = ifelse(newmax, $(Symbol(:i_, d)), $(Symbol(:j_, d))))
            push!(block.args, setj)
        end
        # Push to after reduction loop
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        # Potential loop-carried dependency
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)Iₖ    : I₁ + D₁I₂ + D₁D₂I₃ + ⋯ + D₁D₂⋯Dₖ₋₁Iₖ
        setc = Expr(:call, :+)
        for d ∈ rinds
            push!(setc.args, d == 1 ? :j_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
        end
        for d ∈ nrinds
            push!(setc.args, d == 1 ? :i_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:i_, d)))
        end
        # These complete the expression: 1 + ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)(Iₖ - 1)
        push!(setc.args, 1, :Dstar)
        postj = Expr(:(=), Cᵥ′, setc)
        push!(rblock.args, postj)
        # strides, offsets
        td = Expr(:tuple)
        for d = 1:N
            push!(td.args, Symbol(:D_, d))
        end
        sz = Expr(:(=), td, Expr(:call, :size, :A_1))
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)    : 1 + D₁ + D₁D₂ + ⋯ + D₁D₂⋯Dₖ₋₁
        dstar = Expr(:call, :+, 1)
        for d = 2:N
            push!(dstar.args, d == 2 ? :D_1 : Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
        end
        return quote
            $t = As
            $sz
            Dstar = $dstar
            Dstar = -Dstar
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            @turbo $loops
            return B, C
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        j = Expr(:tuple)
        for d = 1:N
            push!(j.args, Symbol(:j_, d))
        end
        js = :($j = $(ntuple(_ -> 1, Val(N))))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = indices($t, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        cmpr = Expr(:(=), :newmax, Expr(:call, Symbol(OP.instance), f, :ξ))
        push!(block.args, cmpr)
        setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax, f, :ξ))
        push!(block.args, setmax)
        for d ∈ rinds
            setj = Expr(:(=), Symbol(:j_, d),
                        Expr(:call, :ifelse, :newmax, Symbol(:i_, d), Symbol(:j_, d)))
            # setj = :($(Symbol(:j_, d)) = ifelse(newmax, $(Symbol(:i_, d)), $(Symbol(:j_, d))))
            push!(block.args, setj)
        end
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)Iₖ    : I₁ + D₁I₂ + D₁D₂I₃ + ⋯ + D₁D₂⋯Dₖ₋₁Iₖ
        setc = Expr(:call, :+)
        for d ∈ rinds
            push!(setc.args, d == 1 ? :j_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
        end
        for d ∈ nrinds
            push!(setc.args, d == 1 ? :i_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:i_, d)))
        end
        # These complete the expression: 1 + ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)(Iₖ - 1)
        push!(setc.args, 1, :Dstar)
        # strides, offsets
        td = Expr(:tuple)
        for d = 1:N
            push!(td.args, Symbol(:D_, d))
        end
        sz = Expr(:(=), td, Expr(:call, :size, :A_1))
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)    : 1 + D₁ + D₁D₂ + ⋯ + D₁D₂⋯Dₖ₋₁
        dstar = Expr(:call, :+, 1)
        for d = 2:N
            push!(dstar.args, d == 2 ? :D_1 : Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
        end
        return quote
            $t = As
            $js
            $sz
            Dstar = $dstar
            Dstar = -Dstar
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            $ξ
            @turbo $loops
            Bᵥ[] = ξ
            Cᵥ[] = $setc
            return B, C
        end
    end
end

function branches_findminmax_vararg_quote(OP, I, N::Int, M::Int, P::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vfindminmax_vararg!(f, op, init, B, C, As, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vfindminmax_vararg!(f, op, init, B, C, As, $tc))))
            return q
        end
    end
    return staticdim_findminmax_vararg_quote(OP, I, static_dims, N, P)
end

@generated function _vfindminmax_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₗ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, I, Tₒ, Tₗ, N, P, M, D<:Tuple{Vararg{Integer, M}}}
    branches_findminmax_vararg_quote(OP, I, N, M, P, D)
end
# In the case of rinds = ∅, this just corresponds to a map
@generated function _vfindminmax_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₗ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, Tₗ, N, P}
    :(vvmap!(f, B, As); copyto!(C, LinearIndices(As[1])); return B, C)
end

############################################################################################

# Necessary dispatches to define interface
vtfindmax(f::F, As::Vararg{AbstractArray, P}) where {F<:Function, P} =
    vtfindminmax(f, >, typemin, As, :)
vtfindmin(f::F, As::Vararg{AbstractArray, P}) where {F<:Function, P} =
    vtfindminmax(f, <, typemax, As, :)

vtfindmax(f, As::Vararg{AbstractArray, P}; dims=:) where {P} =
    vtfindminmax(f, >, typemin, As, dims)
vtfindmin(f, As::Vararg{AbstractArray, P}; dims=:) where {P} =
    vtfindminmax(f, <, typemax, As, dims)

function vtfindminmax(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, I, M, P}
    A₁ = As[1]
    ax = axes(A₁)
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴬ = size(A₁)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], ndims(A₁))
    B = similar(A₁, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Dᴮ′)
    C = similar(A₁, Int, Dᴮ′)
    _vtfindminmax_vararg!(f, op, init, B, C, As, dims)
    return B, CartesianIndices(A₁)[C]
end

# Over all dims, i.e. if dims = :
function vtfindminmax(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, ::Colon) where {F, OP, I, P}
    _tfindminmaxall_vararg(f, op, init, As)
end
function vtfindminmax(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractVector, P}}, ::Colon) where {F, OP, I, P}
    ξ, CI = _tfindminmaxall_vararg(f, op, init, As)
    return ξ, CI[1]
end

function tfindminmaxall_vararg_quote(OP, I, P::Int)
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
    cmpr = Expr(:(=), :newmax, Expr(:call, Symbol(OP.instance), f, :ξ))
    push!(block.args, cmpr)
    setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax, f, :ξ))
    push!(block.args, setmax)
    setindmax = Expr(:(=), :indmax, Expr(:call, :ifelse, :newmax, :i, :indmax))
    push!(block.args, setindmax)
    # Pre-reduction
    ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance),
                            :(Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)))))
    return quote
        $t = As
        indmax = 1
        $ξ
        @tturbo $loops
        return ξ, CartesianIndices(A_1)[indmax]
    end
end

@generated function _tfindminmaxall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, I, P}
    tfindminmaxall_vararg_quote(OP, I, P)
end

function staticdim_tfindminmax_vararg_quote(OP, I, static_dims::Vector{Int}, N::Int, P::Int)
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
    Cᵥ = Expr(:call, :view, :C)
    Bᵥ′ = Expr(:ref, :Bᵥ)
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
        tc = copy(t)
        push!(tc.args, :B)
        push!(tc.args, :C)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices($tc, $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($tc, $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        push!(rblock.args, ξ)
        for d ∈ rinds
            push!(rblock.args, Expr(:(=), Symbol(:j_, d), Expr(:call, :one, :Int)))
        end
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        cmpr = Expr(:(=), :newmax, Expr(:call, Symbol(OP.instance), f, :ξ))
        push!(block.args, cmpr)
        setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax, f, :ξ))
        push!(block.args, setmax)
        for d ∈ rinds
            setj = Expr(:(=), Symbol(:j_, d),
                        Expr(:call, :ifelse, :newmax, Symbol(:i_, d), Symbol(:j_, d)))
            # setj = :($(Symbol(:j_, d)) = ifelse(newmax, $(Symbol(:i_, d)), $(Symbol(:j_, d))))
            push!(block.args, setj)
        end
        # Push to after reduction loop
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        # Potential loop-carried dependency
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)Iₖ    : I₁ + D₁I₂ + D₁D₂I₃ + ⋯ + D₁D₂⋯Dₖ₋₁Iₖ
        setc = Expr(:call, :+)
        for d ∈ rinds
            push!(setc.args, d == 1 ? :j_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
        end
        for d ∈ nrinds
            push!(setc.args, d == 1 ? :i_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:i_, d)))
        end
        # These complete the expression: 1 + ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)(Iₖ - 1)
        push!(setc.args, 1, :Dstar)
        postj = Expr(:(=), Cᵥ′, setc)
        push!(rblock.args, postj)
        # strides, offsets
        td = Expr(:tuple)
        for d = 1:N
            push!(td.args, Symbol(:D_, d))
        end
        sz = Expr(:(=), td, Expr(:call, :size, :A_1))
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)    : 1 + D₁ + D₁D₂ + ⋯ + D₁D₂⋯Dₖ₋₁
        dstar = Expr(:call, :+, 1)
        for d = 2:N
            push!(dstar.args, d == 2 ? :D_1 : Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
        end
        return quote
            $t = As
            $sz
            Dstar = $dstar
            Dstar = -Dstar
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            @tturbo $loops
            return B, C
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        j = Expr(:tuple)
        for d = 1:N
            push!(j.args, Symbol(:j_, d))
        end
        js = :($j = $(ntuple(_ -> 1, Val(N))))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = indices($t, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices($t, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        cmpr = Expr(:(=), :newmax, Expr(:call, Symbol(OP.instance), f, :ξ))
        push!(block.args, cmpr)
        setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax, f, :ξ))
        push!(block.args, setmax)
        for d ∈ rinds
            setj = Expr(:(=), Symbol(:j_, d),
                        Expr(:call, :ifelse, :newmax, Symbol(:i_, d), Symbol(:j_, d)))
            # setj = :($(Symbol(:j_, d)) = ifelse(newmax, $(Symbol(:i_, d)), $(Symbol(:j_, d))))
            push!(block.args, setj)
        end
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)Iₖ    : I₁ + D₁I₂ + D₁D₂I₃ + ⋯ + D₁D₂⋯Dₖ₋₁Iₖ
        setc = Expr(:call, :+)
        for d ∈ rinds
            push!(setc.args, d == 1 ? :j_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
        end
        for d ∈ nrinds
            push!(setc.args, d == 1 ? :i_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:i_, d)))
        end
        # These complete the expression: 1 + ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)(Iₖ - 1)
        push!(setc.args, 1, :Dstar)
        # strides, offsets
        td = Expr(:tuple)
        for d = 1:N
            push!(td.args, Symbol(:D_, d))
        end
        sz = Expr(:(=), td, Expr(:call, :size, :A_1))
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)    : 1 + D₁ + D₁D₂ + ⋯ + D₁D₂⋯Dₖ₋₁
        dstar = Expr(:call, :+, 1)
        for d = 2:N
            push!(dstar.args, d == 2 ? :D_1 : Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
        end
        return quote
            $t = As
            $js
            $sz
            Dstar = $dstar
            Dstar = -Dstar
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            $ξ
            @tturbo $loops
            Bᵥ[] = ξ
            Cᵥ[] = $setc
            return B, C
        end
    end
end

function branches_tfindminmax_vararg_quote(OP, I, N::Int, M::Int, P::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtfindminmax_vararg!(f, op, init, B, C, As, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vtfindminmax_vararg!(f, op, init, B, C, As, $tc))))
            return q
        end
    end
    return staticdim_tfindminmax_vararg_quote(OP, I, static_dims, N, P)
end

@generated function _vtfindminmax_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₗ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, I, Tₒ, Tₗ, N, P, M, D<:Tuple{Vararg{Integer, M}}}
    branches_tfindminmax_vararg_quote(OP, I, N, M, P, D)
end
# In the case of rinds = ∅, this just corresponds to a map
@generated function _vtfindminmax_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₗ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, Tₗ, N, P}
    :(vtmap!(f, B, As); copyto!(C, LinearIndices(As[1])); return B, C)
end

############################################################################################
