#
# Date created: 2022-04-01
# Author: aradclif
#
#
############################################################################################
# Revision of interface: in fact, the interface is complicated more by the introduction of
# `vvmapreduce_vararg`, as it could have been handled by simply dispatching from `vvmapreduce`.
# It does seem worthwhile to use separate generated functions, but that is an internal
# detail of the `vvmapreduce` dispatch.
# Also, it seems that the specialization on Vararg{S, P} where S<:AbstractArray{T, N}
# is unnecessary at both the public-facing and generated function interfaces.
# Moreover, based on a variety of test cases, it seems to provide no benefits
# to performance. Given that the generated function branching would theoretically
# be specialized as well, it would certainly lead to longer compilation times, and, perhaps
# a small performance hit if one were to expand a massive number of the branches?
# -- doubtful, but one should be wary of combinatorial explosions.
# I leave the specialization code in-place, but commented out. If a future use warrants it,
# it can simply be reactivated.

"""
    vvmapreduce(f, op, init, As::Vararg{AbstractArray, N}) where {N}

Version of mapreduce for `f` : ℝᴺ → ℝ, with reduction occurring over all dimensions.
"""
vvmapreduce(f::F, op::OP, init::I, As::Vararg{AbstractArray, P}) where {F, OP, I, P} =
    vvmapreduce(f, op, init, As, :)

"""
    vvmapreduce(f, op, As::Vararg{AbstractArray, N}; dims=:, init) where {N}

Keyword args version for `f` : ℝᴺ → ℝ.
"""
vvmapreduce(f, op, As::Vararg{AbstractArray, P}; dims=:, init) where {P} =
    vvmapreduce(f, op, init, As, dims)

# While this is a nice convenience, it does add 16 dispatches to the user-visible method table.
# On the other hand, it does not cause any issues.
for (op, init) ∈ zip((:+, :*, :max,:min), (:zero, :one, :typemin, :typemax))
    # Convenience default initializers.
    # First line covers both AbstractArray and ::Tuple{Vararg{AbstractArray, P}}
    @eval vvmapreduce(f, ::typeof($op), A; dims=:, init=$init) = vvmapreduce(f, $op, init, A, dims)
    @eval vvmapreduce(f, ::typeof($op), As::Vararg{AbstractArray, P}; dims=:, init=$init) where {P} = vvmapreduce(f, $op, init, As, dims)
    # 3-argument versions for common binary ops
    @eval vvmapreduce(f::F, ::typeof($op), A) where {F<:Function} = vvmapreduce(f, $op, $init, A, :)
    @eval vvmapreduce(f::F, ::typeof($op), As::Vararg{AbstractArray, P}) where {F<:Function, P} = vvmapreduce(f, $op, $init, As, :)
end

################
# function vvmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M, S<:AbstractArray{T, N}, P}
#     ax = axes(As[1])
#     for p = 2:P
#         axes(As[p]) == ax || throw(DimensionMismatch)
#     end
#     Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), Val(N))
#     B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(_ -> T, Val(P))...), Int), Dᴮ′)
#     _vvmapreduce_vararg!(f, op, init, B, As, dims)
# end
# One approach to handle differently typed arrays is have an additional method as below,
# and to provide generated functions that also accept Vararg{AbstractArray}

"""
    vvmapreduce(f, op, init, As::Tuple{Vararg{AbstractArray}}, dims=:)

Version of mapreduce for `f` : ℝᴺ → ℝ, with reduction over given `dims`.
"""
function vvmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, I, M, P}
    ax = axes(As[1])
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), ndims(As[1]))
    B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int), Dᴮ′)
    _vvmapreduce_vararg!(f, op, init, B, As, dims)
    return B
end

function staticdim_mapreduce_vararg_quote(OP, I, static_dims::Vector{Int}, N::Int, P::Int)
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
        loops = Expr(:for, Expr(:(=), Symbol(:i_, nrinds[1]),
                                Expr(:call, :indices, tc, nrinds[1])), block)
        for i = 2:length(nrinds)
            newblock = Expr(:block)
            push!(block.args,
                  Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
                                  Expr(:call, :indices, tc, nrinds[i])), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d),
                                              Expr(:call, :indices, t, d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            $t = As
            Bᵥ = $Bᵥ
            @turbo check_empty=true $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, Expr(:(=), Symbol(:i_, rinds[1]), Expr(:call, :indices, t, rinds[1])), block)
        for i = 2:length(rinds)
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                                              Expr(:call, :indices, t, rinds[i])), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        return quote
            $t = As
            Bᵥ = $Bᵥ
            $ξ
            @turbo check_empty=true $loops
            Bᵥ[] = ξ
            return B
        end
    end
end

function branches_mapreduce_vararg_quote(OP, I, N::Int, M::Int, P::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vvmapreduce_vararg!(f, op, init, B, As, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vvmapreduce_vararg!(f, op, init, B, As, $tc))))
            return q
        end
    end
    return staticdim_mapreduce_vararg_quote(OP, I, static_dims, N, P)
end

# In the case of rinds = ∅, this just corresponds to a map
function map_vararg_quote(P::Int)
    # t = Expr(:tuple)
    # for p = 1:P
    #     push!(t.args, Symbol(:A_, p))
    # end
    # f = Expr(:call, :f)
    # for p = 1:P
    #     A = Expr(:ref, Symbol(:A_, p), ntuple(d -> Symbol(:i_, d), N)...)
    #     push!(f.args, A)
    # end
    # tc = copy(t)
    # push!(tc.args, :B)
    # block = Expr(:block)
    # loops = Expr(:for, Expr(:(=), Symbol(:i_, N), Expr(:call, :indices, tc, N)), block)
    # for d = N-1:-1:1
    #     newblock = Expr(:block)
    #     push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :indices, tc, d)), newblock))
    #     block = newblock
    # end
    # # Push to inside innermost loop
    # setb = Expr(:(=), Expr(:ref, :B, ntuple(d -> Symbol(:i_, d), N)...), f)
    # push!(block.args, setb)
    # Version using eachindex
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
    setb = Expr(:(=), Expr(:ref, :B, :i), f)
    push!(block.args, setb)
    return quote
        $t = As
        @turbo check_empty=true $loops
        return B
    end
end

# @generated function _vvmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::D) where {F, OP, I, Tₒ, T, N, M, S<:AbstractArray{T, N}, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
#     branches_mapreduce_vararg_quote(OP, I, N, M, P, D)
# end

# Should technically just be map, as it is the case of rinds = ∅
# @generated function _vvmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N, S<:AbstractArray{T, N}, P}
#     map_vararg_quote(P)
# end

# Versions which cover arrays with different element types
@generated function _vvmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, I, Tₒ, N, M, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_mapreduce_vararg_quote(OP, I, N, M, P, D)
end

@generated function _vvmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, N, P}
    map_vararg_quote(P)
end

# Reduction over all dims, i.e. if dims = :
# Rather than check the dimensions, just leave it to `eachindex` to throw --
# it gives an informative message, and there's no need to repeat it.
# function vvmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, ::Colon) where {F, OP, I, T, N, S<:AbstractArray{T, N}, P}
#     _mapreduceall_vararg(f, op, init, As)
# end
function vvmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, ::Colon) where {F, OP, I, P}
    _mapreduceall_vararg(f, op, init, As)
end

function mapreduceall_vararg_quote(OP, I, P::Int)
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
        @turbo check_empty=true $loops
        return ξ
    end
end

# @generated function _mapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}) where {F, OP, I, T, N, S<:AbstractArray{T, N}, P}
#     mapreduceall_vararg_quote(OP, I, P)
# end

@generated function _mapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, I, P}
    mapreduceall_vararg_quote(OP, I, P)
end

################
# Version wherein an initial value is supplied
# These functions could be eliminated by dispatching on the generated functions,
# provided the generated functions matched the name _vvmapreduce_vararg!.
# TO DO: test performance of above proposal.

# function vvmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, dims::NTuple{M, Int}) where {F, OP, I<:Number, T, N, M, S<:AbstractArray{T, N}, P}
#     ax = axes(As[1])
#     for p = 2:P
#         axes(As[p]) == ax || throw(DimensionMismatch)
#     end
#     Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), Val(N))
#     B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(_ -> T, Val(P))...), Int), Dᴮ′)
#     _vvmapreduce_vararg_init!(f, op, init, B, As, dims)
#     return B
# end
function vvmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, I<:Number, M, P}
    ax = axes(As[1])
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), ndims(As[1]))
    B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int), Dᴮ′)
    _vvmapreduce_vararg_init!(f, op, init, B, As, dims)
end

function staticdim_mapreduce_vararg_init_quote(OP, static_dims::Vector{Int}, N::Int, P::Int)
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
        ξ₀ = Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :init)
        block = Expr(:block)
        tc = copy(t)
        push!(tc.args, :B)
        loops = Expr(:for, Expr(:(=), Symbol(:i_, nrinds[1]),
                                Expr(:call, :indices, tc, nrinds[1])), block)
        for i = 2:length(nrinds)
            newblock = Expr(:block)
            push!(block.args,
                  Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
                                  Expr(:call, :indices, tc, nrinds[i])), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, :ξ₀)
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d),
                                              Expr(:call, :indices, t, d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            $t = As
            Bᵥ = $Bᵥ
            ξ₀ = $ξ₀
            @turbo check_empty=true $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :init))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, Expr(:(=), Symbol(:i_, rinds[1]), Expr(:call, :indices, t, rinds[1])), block)
        for i = 2:length(rinds)
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                                              Expr(:call, :indices, t, rinds[i])), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        return quote
            $t = As
            Bᵥ = $Bᵥ
            $ξ
            @turbo check_empty=true $loops
            Bᵥ[] = ξ
            return B
        end
    end
end

function branches_mapreduce_vararg_init_quote(OP, N::Int, M::Int, P::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vvmapreduce_vararg_init!(f, op, init, B, As, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vvmapreduce_vararg_init!(f, op, init, B, As, $tc))))
            return q
        end
    end
    return staticdim_mapreduce_vararg_init_quote(OP, static_dims, N, P)
end

# @generated function _vvmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::D) where {F, OP, I, Tₒ, T, N, M, S<:AbstractArray{T, N}, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
#     branches_mapreduce_vararg_init_quote(OP, N, M, P, D)
# end
# @generated function _vvmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N, S<:AbstractArray{T, N}, P}
#     map_vararg_quote(P)
# end

# Versions which cover arrays with different element types
@generated function _vvmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, I, Tₒ, N, M, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_mapreduce_vararg_init_quote(OP, N, M, P, D)
end

@generated function _vvmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, N, P}
    map_vararg_quote(P)
end

# Reduction over all dims, i.e. if dims = :
# Rather than check the dimensions, just leave it to `eachindex` to throw --
# it gives an informative message, and there's no need to repeat it.
# function vvmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, ::Colon) where {F, OP, I<:Number,T, N, S<:AbstractArray{T, N}, P}
#     _mapreduceall_vararg(f, op, init, As)
# end
# function vvmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, ::Colon) where {F, OP, I<:Number, M, P}
#     _mapreduceall_vararg(f, op, init, As)
# end

function mapreduceall_vararg_init_quote(OP, P::Int)
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
        @turbo check_empty=true $loops
        return ξ
    end
end

# @generated function _mapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}) where {F, OP, I<:Number, T, N, S<:AbstractArray{T, N}, P}
#     mapreduceall_vararg_init_quote(OP, P)
# end

@generated function _mapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, I<:Number, P}
    mapreduceall_vararg_init_quote(OP, P)
end

############################################################################################
# Threaded version

vtmapreduce(f::F, op::OP, init::I, As::Vararg{AbstractArray, P}) where {F, OP, I, P} = vtmapreduce(f, op, init, As, :)

"""
    vtmapreduce(f, op, init, As::Vararg{AbstractArray, N}; dims=:, init) where {N}

Version for `f` : ℝᴺ → ℝ, with reduction over `dims`. Threaded.
"""
vtmapreduce(f, op, As::Vararg{AbstractArray, P}; dims=:, init) where {P} = vtmapreduce(f, op, init, As, dims)

for (op, init) ∈ zip((:+, :*, :max,:min), (:zero, :one, :typemin, :typemax))
    # Convenience default initializers.
    # First line covers both AbstractArray and ::Tuple{Vararg{AbstractArray, P}}
    @eval vtmapreduce(f, ::typeof($op), A; dims=:, init=$init) = vtmapreduce(f, $op, init, A, dims)
    @eval vtmapreduce(f, ::typeof($op), As::Vararg{AbstractArray, P}; dims=:, init=$init) where {P} = vtmapreduce(f, $op, init, As, dims)
    # 3-argument versions for common binary ops
    @eval vtmapreduce(f::F, ::typeof($op), A) where {F<:Function} = vtmapreduce(f, $op, $init, A, :)
    @eval vtmapreduce(f::F, ::typeof($op), As::Vararg{AbstractArray, P}) where {F<:Function, P} = vtmapreduce(f, $op, $init, As, :)
end

################
# function vtmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M, S<:AbstractArray{T, N}, P}
#     ax = axes(As[1])
#     for p = 2:P
#         axes(As[p]) == ax || throw(DimensionMismatch)
#     end
#     Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), Val(N))
#     B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(_ -> T, Val(P))...), Int), Dᴮ′)
#     _vtmapreduce_vararg!(f, op, init, B, As, dims)
# end
# One approach to handle differently typed arrays is have an additional method as below,
# and to provide generated functions that also accept Vararg{AbstractArray}
"""
    vtmapreduce(f, op, init, As::Tuple{Vararg{AbstractArray}}, dims=:)

Version for `f` : ℝᴺ → ℝ, with reduction over `dims`. Threaded.
"""
function vtmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, I, M, P}
    ax = axes(As[1])
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), ndims(As[1]))
    B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int), Dᴮ′)
    _vtmapreduce_vararg!(f, op, init, B, As, dims)
    return B
end

function staticdim_tmapreduce_vararg_quote(OP, I, static_dims::Vector{Int}, N::Int, P::Int)
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
        loops = Expr(:for, Expr(:(=), Symbol(:i_, nrinds[1]),
                                Expr(:call, :indices, tc, nrinds[1])), block)
        for i = 2:length(nrinds)
            newblock = Expr(:block)
            push!(block.args,
                  Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
                                  Expr(:call, :indices, tc, nrinds[i])), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d),
                                              Expr(:call, :indices, t, d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            $t = As
            Bᵥ = $Bᵥ
            @tturbo check_empty=true $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, Expr(:(=), Symbol(:i_, rinds[1]), Expr(:call, :indices, t, rinds[1])), block)
        for i = 2:length(rinds)
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                                              Expr(:call, :indices, t, rinds[i])), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        return quote
            $t = As
            Bᵥ = $Bᵥ
            $ξ
            @tturbo check_empty=true $loops
            Bᵥ[] = ξ
            return B
        end
    end
end

function branches_tmapreduce_vararg_quote(OP, I, N::Int, M::Int, P::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtmapreduce_vararg!(f, op, init, B, As, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vtmapreduce_vararg!(f, op, init, B, As, $tc))))
            return q
        end
    end
    return staticdim_tmapreduce_vararg_quote(OP, I, static_dims, N, P)
end

# In the case of rinds = ∅, this just corresponds to a map
function tmap_vararg_quote(P::Int)
    # t = Expr(:tuple)
    # for p = 1:P
    #     push!(t.args, Symbol(:A_, p))
    # end
    # f = Expr(:call, :f)
    # for p = 1:P
    #     A = Expr(:ref, Symbol(:A_, p), ntuple(d -> Symbol(:i_, d), N)...)
    #     push!(f.args, A)
    # end
    # tc = copy(t)
    # push!(tc.args, :B)
    # block = Expr(:block)
    # loops = Expr(:for, Expr(:(=), Symbol(:i_, N), Expr(:call, :indices, tc, N)), block)
    # for d = N-1:-1:1
    #     newblock = Expr(:block)
    #     push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :indices, tc, d)), newblock))
    #     block = newblock
    # end
    # # Push to inside innermost loop
    # setb = Expr(:(=), Expr(:ref, :B, ntuple(d -> Symbol(:i_, d), N)...), f)
    # push!(block.args, setb)
    # Version using eachindex
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
    setb = Expr(:(=), Expr(:ref, :B, :i), f)
    push!(block.args, setb)
    return quote
        $t = As
        @tturbo check_empty=true $loops
        return B
    end
end

# @generated function _vtmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::D) where {F, OP, I, Tₒ, T, N, M, S<:AbstractArray{T, N}, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
#     branches_tmapreduce_vararg_quote(OP, I, N, M, P, D)
# end

# Should technically just be map, as it is the case of rinds = ∅
# @generated function _vtmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N, S<:AbstractArray{T, N}, P}
#     tmap_vararg_quote(P)
# end

# Versions which cover arrays with different element types
@generated function _vtmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, I, Tₒ, N, M, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_tmapreduce_vararg_quote(OP, I, N, M, P, D)
end

@generated function _vtmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, N, P}
    tmap_vararg_quote(P)
end

# Reduction over all dims, i.e. if dims = :
# Rather than check the dimensions, just leave it to `eachindex` to throw --
# it gives an informative message, and there's no need to repeat it.
# function vtmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, ::Colon) where {F, OP, I, T, N, S<:AbstractArray{T, N}, P}
#     _tmapreduceall_vararg(f, op, init, As)
# end
function vtmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, ::Colon) where {F, OP, I, P}
    _tmapreduceall_vararg(f, op, init, As)
end

function tmapreduceall_vararg_quote(OP, I, P::Int)
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
        @tturbo check_empty=true $loops
        return ξ
    end
end

# @generated function _tmapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}) where {F, OP, I, T, N, S<:AbstractArray{T, N}, P}
#     tmapreduceall_vararg_quote(OP, I, P)
# end

@generated function _tmapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, I, P}
    tmapreduceall_vararg_quote(OP, I, P)
end

################
# function vtmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, dims::NTuple{M, Int}) where {F, OP, I<:Number, T, N, M, S<:AbstractArray{T, N}, P}
#     ax = axes(As[1])
#     for p = 2:P
#         axes(As[p]) == ax || throw(DimensionMismatch)
#     end
#     Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), Val(N))
#     B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(_ -> T, Val(P))...), Int), Dᴮ′)
#     _vtmapreduce_vararg_init!(f, op, init, B, As, dims)
#     return B
# end
function vtmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, I<:Number, M, P}
    ax = axes(As[1])
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), ndims(As[1]))
    B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int), Dᴮ′)
    _vtmapreduce_vararg_init!(f, op, init, B, As, dims)
end

function staticdim_tmapreduce_vararg_init_quote(OP, static_dims::Vector{Int}, N::Int, P::Int)
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
        ξ₀ = Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :init)
        block = Expr(:block)
        tc = copy(t)
        push!(tc.args, :B)
        loops = Expr(:for, Expr(:(=), Symbol(:i_, nrinds[1]),
                                Expr(:call, :indices, tc, nrinds[1])), block)
        for i = 2:length(nrinds)
            newblock = Expr(:block)
            push!(block.args,
                  Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
                                  Expr(:call, :indices, tc, nrinds[i])), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, :ξ₀)
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d),
                                              Expr(:call, :indices, t, d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            $t = As
            Bᵥ = $Bᵥ
            ξ₀ = $ξ₀
            @tturbo check_empty=true $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :init))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, Expr(:(=), Symbol(:i_, rinds[1]), Expr(:call, :indices, t, rinds[1])), block)
        for i = 2:length(rinds)
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                                              Expr(:call, :indices, t, rinds[i])), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, f))
        push!(block.args, setξ)
        return quote
            $t = As
            Bᵥ = $Bᵥ
            $ξ
            @tturbo check_empty=true $loops
            Bᵥ[] = ξ
            return B
        end
    end
end

function branches_tmapreduce_vararg_init_quote(OP, N::Int, M::Int, P::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtmapreduce_vararg_init!(f, op, init, B, As, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vtmapreduce_vararg_init!(f, op, init, B, As, $tc))))
            return q
        end
    end
    return staticdim_tmapreduce_vararg_init_quote(OP, static_dims, N, P)
end

# @generated function _vtmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::D) where {F, OP, I, Tₒ, T, N, M, S<:AbstractArray{T, N}, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
#     branches_tmapreduce_vararg_init_quote(OP, N, M, P, D)
# end
# @generated function _vtmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N, S<:AbstractArray{T, N}, P}
#     tmap_vararg_quote(P)
# end

# Versions which cover arrays with different element types
@generated function _vtmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, I, Tₒ, N, M, P, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_tmapreduce_vararg_init_quote(OP, N, M, P, D)
end

@generated function _vtmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, N, P}
    tmap_vararg_quote(P)
end

# function vtmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, ::Colon) where {F, OP, I<:Number,T, N, S<:AbstractArray{T, N}, P}
#     _tmapreduceall_vararg(f, op, init, As)
# end
# function vtmapreduce(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, ::Colon) where {F, OP, I<:Number, M, P}
#     _tmapreduceall_vararg(f, op, init, As)
# end

function tmapreduceall_vararg_init_quote(OP, P::Int)
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
        @tturbo check_empty=true $loops
        return ξ
    end
end

# @generated function _tmapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}) where {F, OP, I<:Number, T, N, S<:AbstractArray{T, N}, P}
#     tmapreduceall_vararg_init_quote(OP, P)
# end

@generated function _tmapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, I<:Number, P}
    tmapreduceall_vararg_init_quote(OP, P)
end
