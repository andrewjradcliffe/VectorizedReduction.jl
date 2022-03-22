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

function vvmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    # B = similar(A, Base.promote_op(op, T, Int), Dᴮ′)
    B = similar(A, Base.promote_op(op, Base.promote_op(f, T), Int), Dᴮ′)
    _vvmapreduce!(f, op, init, B, A, dims)
    return B
end

# Convenience definitions
vvsum(f::F, A, dims) where {F} = vvmapreduce(f, +, zero, A, dims)
vvprod(f::F, A, dims) where {F} = vvmapreduce(f, *, one, A, dims)
vvmaximum(f::F, A, dims) where {F} = vvmapreduce(f, max, typemin, A, dims)
vvminimum(f::F, A, dims) where {F} = vvmapreduce(f, min, typemax, A, dims)

vvsum(A, dims) = vvmapreduce(identity, +, zero, A, dims)
vvprod(A, dims) = vvmapreduce(identity, *, one, A, dims)
vvmaximum(A, dims) = vvmapreduce(identity, max, typemin, A, dims)
vvminimum(A, dims) = vvmapreduce(identity, min, typemax, A, dims)

# reduction over all dims
@generated function vvmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I, T, N}
    fsym = F.instance
    opsym = OP.instance
    initsym = I.instance
    Tₒ = Base.promote_op(opsym, Base.promote_op(fsym, T), Int)
    quote
        ξ = $initsym($Tₒ)
        @turbo for i ∈ eachindex(A)
            ξ = $opsym($fsym(A[i]), ξ)
        end
        return ξ
    end
end

vvsum(f::F, A) where {F<:Function} = vvmapreduce(f, +, zero, A, :)
vvprod(f::F, A) where {F<:Function} = vvmapreduce(f, *, one, A, :)
vvmaximum(f::F, A) where {F<:Function} = vvmapreduce(f, max, typemin, A, :)
vvminimum(f::F, A) where {F<:Function} = vvmapreduce(f, min, typemax, A, :)

vvsum(A) = vvmapreduce(identity, +, zero, A, :)
vvprod(A) = vvmapreduce(identity, *, one, A, :)
vvmaximum(A) = vvmapreduce(identity, max, typemin, A, :)
vvminimum(A) = vvmapreduce(identity, min, typemax, A, :)

# A surprising convenience opportunity -- albeit, a specific implementation will
# be necessary in order to utilize Array{Bool} rather than the default, which
# will get promoted to Array{Int}. Oddly, only works on Array{<:Integer} inputs.
# A (inefficient) workaround would be to sum Bools, then compare to length
# vany(f::F, A, dims) where {F} = vvmapreduce(f, |, zero, A, dims)
# vall(f::F, A, dims) where {F} = vvmapreduce(f, &, one, A, dims)
# vany(A, dims) = vvmapreduce(identity, |, zero, A, dims)
# vall(A, dims) = vvmapreduce(identity, &, one, A, dims)

# Define reduce
vvreduce(op::OP, init::I, A, dims) where {OP, I} = vvmapreduce(identity, op, init, A, dims)

function staticdim_mapreduce_quote(F, OP, I, static_dims::Vector{Int}, N::Int)
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
        for i = 2:length(nrinds)
            newblock = Expr(:block)
            push!(block.args,
                  Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
                                  Expr(:call, :indices, Expr(:tuple, :A, :B), nrinds[i])), newblock))
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
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance),
                                   Expr(:call, Symbol(F.instance), A), :ξ))
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
        loops = Expr(:for, Expr(:(=), Symbol(:i_, rinds[1]),
                                Expr(:call, :axes, :A, rinds[1])), block)
        for i = 2:length(rinds)
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                                              Expr(:call, :axes, :A, rinds[i])), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance),
                                   Expr(:call, Symbol(F.instance), A), :ξ))
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

function branches_mapreduce_quote(F, OP, I, N::Int, M::Int, D)
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
    return staticdim_mapreduce_quote(F, OP, I, static_dims, N)
end

@generated function _vvmapreduce!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_mapreduce_quote(F, OP, I, N, M, D)
end
@generated function _vvmapreduce!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N}
    :(copyto!(B, A); return B)
end

