#
# Date created: 2022-04-01
# Author: aradclif
#
#
############################################################################################

function vvmapreduce_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M, S<:AbstractArray{T, N}, P}
    ax = axes(As[1])
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), Val(N))
    B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(_ -> T, Val(P))...), Int), Dᴮ′)
    _vvmapreduce_vararg!(f, op, init, B, As, dims)
end
# One approach to handle differently typed arrays is have an additional method as below,
# and to provide generated functions that also accept Vararg{AbstractArray}
function vvmapreduce_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, I, M, P}
    ax = axes(As[1])
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), ndims(As[1]))
    B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int), Dᴮ′)
    _vvmapreduce_vararg!(f, op, init, B, As, dims)
end

A1 = rand(5,5,5,5);
A2 = rand(5,5,5,5);
A3 = rand(5,5,5,5);
as = (A1, A2, A3);
@benchmark vvmapreduce_vararg(+, +, zero, as, (1,2,4))
@benchmark mapreduce(+, +, A1, A2, A3, dims=(1, 2,4))
vvmapreduce_vararg(+, +, zero, as, (1,2,4)) ≈ mapreduce(+, +, A1, A2, A3, dims=(1, 2,4))
g(x, y, z) = x * y + z
@benchmark vvmapreduce_vararg(g, +, zero, as, (1,2,4))
vvmapreduce_vararg((x, y, z) -> x+y+z, +, zero, as, (1,2,3,4))
vvmapreduce_vararg(+, +, zero, as, (5,)) ≈ mapreduce(+, +, A1, A2, A3, dims=5)
@benchmark vvmapreduce_vararg(+, +, zero, as, (5,))
@benchmark vmap(+, as...)

# Tests of variably typed arrays
A4 = rand(1:5, 5,5,5,5);
@benchmark vvmapreduce_vararg(+, +, zero, (A1, A2), (2,3,4))
@benchmark vvmapreduce_vararg(+, +, zero, (A1, A4), (2,3,4))
vvmapreduce_vararg(+, +, zero, (A1, A4), (2,3,4)) ≈ mapreduce(+, +, A1, A4, dims=(2,3,4))

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
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), f, :ξ))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            $t = As
            Bᵥ = $Bᵥ
            @turbo $loops
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
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), f, :ξ))
        push!(block.args, setξ)
        return quote
            $t = As
            Bᵥ = $Bᵥ
            $ξ
            @turbo $loops
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

@generated function _vvmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::D) where {F, OP, I, Tₒ, T, N, M, S<:AbstractArray{T, N}, P, D<:Tuple{Vararg{Integer, M}}}
    branches_mapreduce_vararg_quote(OP, I, N, M, P, D)
end

# In the case of rinds = ∅, this just corresponds to a map
function map_vararg_quote(N::Int, P::Int)
    t = Expr(:tuple)
    for p = 1:P
        push!(t.args, Symbol(:A_, p))
    end
    f = Expr(:call, :f)
    for p = 1:P
        A = Expr(:ref, Symbol(:A_, p), ntuple(d -> Symbol(:i_, d), N)...)
        push!(f.args, A)
    end
    tc = copy(t)
    push!(tc.args, :B)
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), Symbol(:i_, N), Expr(:call, :indices, tc, N)), block)
    for d = N-1:-1:1
        newblock = Expr(:block)
        push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :indices, tc, d)), newblock))
        block = newblock
    end
    # Push to inside innermost loop
    setb = Expr(:(=), Expr(:ref, :B, ntuple(d -> Symbol(:i_, d), N)...), f)
    push!(block.args, setb)
    return quote
        $t = As
        @turbo $loops
        return B
    end
end

# Should technically just be map, as it is the case of rinds = ∅
@generated function _vvmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N, S<:AbstractArray{T, N}, P}
    map_vararg_quote(N, P)
end

# Versions which cover arrays with different element types
@generated function _vvmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, I, Tₒ, N, M, P, D<:Tuple{Vararg{Integer, M}}}
    branches_mapreduce_vararg_quote(OP, I, N, M, P, D)
end

@generated function _vvmapreduce_vararg!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, N, P}
    map_vararg_quote(N, P)
end

# Reduction over all dims, i.e. if dims = :
# Rather than check the dimensions, just leave it to `eachindex` to throw --
# it gives an informative message, and there's no need to repeat it.
function vvmapreduce_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, ::Colon) where {F, OP, I,T, N, S<:AbstractArray{T, N}, P}
    _mapreduceall_vararg(f, op, init, As)
end
function vvmapreduce_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, ::Colon) where {F, OP, I, M, P}
    _mapreduceall_vararg(f, op, init, As)
end
vvmapreduce_vararg(f, op, init, As) = vvmapreduce_vararg(f, op, init, As, :)

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
    setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), f, :ξ))
    push!(block.args, setξ)
    # Pre-reduction
    ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :(Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int))))
    return quote
        $t = As
        $ξ
        @turbo $loops
        return ξ
    end
end

@generated function _mapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}) where {F, OP, I, T, N, S<:AbstractArray{T, N}, P}
    mapreduceall_vararg_quote(OP, I, P)
end

@generated function _mapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, I, P}
    mapreduceall_vararg_quote(OP, I, P)
end

################
# Version wherein an initial value is supplied

function vvmapreduce_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, dims::NTuple{M, Int}) where {F, OP, I<:Number, T, N, M, S<:AbstractArray{T, N}, P}
    ax = axes(As[1])
    for p = 2:P
        axes(As[p]) == ax || throw(DimensionMismatch)
    end
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), Val(N))
    B = similar(As[1], Base.promote_op(op, Base.promote_op(f, ntuple(_ -> T, Val(P))...), Int), Dᴮ′)
    _vvmapreduce_vararg_init!(f, op, init, B, As, dims)
    return B
end
function vvmapreduce_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {F, OP, I<:Number, M, P}
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
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), f, :ξ))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            $t = As
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
        loops = Expr(:for, Expr(:(=), Symbol(:i_, rinds[1]), Expr(:call, :indices, t, rinds[1])), block)
        for i = 2:length(rinds)
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                                              Expr(:call, :indices, t, rinds[i])), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), f, :ξ))
        push!(block.args, setξ)
        return quote
            $t = As
            Bᵥ = $Bᵥ
            $ξ
            @turbo $loops
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

@generated function _vvmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::D) where {F, OP, I, Tₒ, T, N, M, S<:AbstractArray{T, N}, P, D<:Tuple{Vararg{Integer, M}}}
    branches_mapreduce_vararg_init_quote(OP, N, M, P, D)
end
@generated function _vvmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{S, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N, S<:AbstractArray{T, N}, P}
    map_vararg_quote(N, P)
end

# Versions which cover arrays with different element types
@generated function _vvmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::D) where {F, OP, I, Tₒ, N, M, P, D<:Tuple{Vararg{Integer, M}}}
    branches_mapreduce_vararg_init_quote(OP, N, M, P, D)
end

@generated function _vvmapreduce_vararg_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, As::Tuple{Vararg{AbstractArray, P}}, dims::Tuple{}) where {F, OP, I, Tₒ, N, P}
    map_vararg_quote(N, P)
end

# Reduction over all dims, i.e. if dims = :
# Rather than check the dimensions, just leave it to `eachindex` to throw --
# it gives an informative message, and there's no need to repeat it.
# function vvmapreduce_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}, ::Colon) where {F, OP, I<:Number,T, N, S<:AbstractArray{T, N}, P}
#     _mapreduceall_vararg(f, op, init, As)
# end
# function vvmapreduce_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}, ::Colon) where {F, OP, I<:Number, M, P}
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
    setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), f, :ξ))
    push!(block.args, setξ)
    # Pre-reduction
    ξ = Expr(:(=), :ξ, Expr(:call, :convert, :(Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, $(ntuple(p -> Expr(:call, :eltype, Symbol(:A_, p)), P)...)), Int)), :init))
    return quote
        $t = As
        $ξ
        @turbo $loops
        return ξ
    end
end

@generated function _mapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{S, P}}) where {F, OP, I<:Number, T, N, S<:AbstractArray{T, N}, P}
    mapreduceall_vararg_init_quote(OP, P)
end

@generated function _mapreduceall_vararg(f::F, op::OP, init::I, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, I<:Number, P}
    mapreduceall_vararg_init_quote(OP, P)
end
