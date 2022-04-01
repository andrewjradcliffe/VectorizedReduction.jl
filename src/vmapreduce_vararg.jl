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
        # loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B), $(nrinds[1]))), block)
        for i = 2:length(nrinds)
            # for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args,
                  Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
                                  Expr(:call, :indices, tc, nrinds[i])), newblock))
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
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d),
                                              Expr(:call, :indices, t, d)), newblock))
            # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
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
        # loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for i = 2:length(rinds)
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                                              Expr(:call, :indices, t, rinds[i])), newblock))
            # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
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
