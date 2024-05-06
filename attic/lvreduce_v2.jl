#
# Date created: 2022-03-20
# Author: aradclif
#
#
############################################################################################
function reduce_quote3(OP, I, N::Int, D)
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    bᵥ = Expr(:call, :view, :B)
    bᵥ′ = Expr(:ref, :Bᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if D.parameters[d] === Static.One
            push!(bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(rinds, d)
        else
            push!(bᵥ.args, :)
            push!(nrinds, d)
            push!(bᵥ′.args, Symbol(:i_, d))
        end
    end
    bᵥ = Expr(:(=), :Bᵥ, bᵥ)
    sort!(rinds, rev=true)
    sort!(nrinds, rev=true)
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
    # Push to before reduction loop
    pre = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
    push!(rblock.args, pre)
    # Reduction loop
    for i = 1:length(rinds)
        newblock = Expr(:block)
        push!(block.args,
              Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                              Expr(:call, :axes, :A, rinds[i])), newblock))
        block = newblock
    end
    # Push to inside innermost loop
    reduction = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, a))
    push!(block.args, reduction)
    # Push to after reduction loop
    post = Expr(:(=), bᵥ′, :ξ)
    push!(rblock.args, post)
    return quote
        $bᵥ
        @tturbo check_empty=true $loops
        return B
    end
end

function _lvreduce3(op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {OP, I, T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        B = hvncat(ntuple(_ -> 1, Val(N)), true, lvreduce1(op, A))
    else
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), Val(N))
        B = similar(A, Base.promote_op(op, T, Int), Dᴮ′)
        _lvreduce3!(op, init, B, A, Dᴮ′)
    end
    return B
end

@generated function _lvreduce3!(op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {OP, I, Tₒ, T, N, D}
    reduce_quote3(OP, I, N, D)
end


# Handle scalar dims by wrapping in Tuple
_lvreduce3(op, A, dims::Int) = _lvreduce3(op, A, (dims,))
# Convenience dispatches to match JuliaBase
lvreduce3(op, A) = lvreduce1(op, A)
_lvreduce3(op, A, ::Colon) = lvreduce1(op, A)

# Common interface for everything related to reduce
function lvreduce3(op::OP, init::I, A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {OP, I, T, N}
    if (multithreaded === :auto && length(A) > 4095) || multithreaded === true
        # B = init === nothing ? _lvtreduce(op, A, dims) : _lvtreduce_init(op, A, dims, init)
        B = _lvreduce3(op, init, A, dims)
    else
        # B = init === nothing ? _lvreduce(op, A, dims) : _lvreduce_init(op, A, dims, init)
        B = _lvreduce3(op, init, A, dims)
    end
    return B
end

# Convenience definitions
lvsum3(A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N} =
    lvreduce3(+, zero, A, dims=dims, multithreaded=multithreaded)
lvprod3(A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N} =
    lvreduce3(*, one, A, dims=dims, multithreaded=multithreaded)
lvmaximum3(A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N} =
    lvreduce3(max, typemin, A, dims=dims, multithreaded=multithreaded)
lvminimum3(A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N} =
    lvreduce3(min, typemax, A, dims=dims, multithreaded=multithreaded)
lvextrema3(A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N} =
    collect(zip(lvminimum3(A, dims=dims, multithreaded=multithreaded),
                lvmaximum3(A, dims=dims, multithreaded=multithreaded)))

################
function mapreduce_quote3(F, OP, I, N::Int, D)
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    bᵥ = Expr(:call, :view, :B)
    bᵥ′ = Expr(:ref, :Bᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if D.parameters[d] === Static.One
            push!(bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(rinds, d)
        else
            push!(bᵥ.args, :)
            push!(nrinds, d)
            push!(bᵥ′.args, Symbol(:i_, d))
        end
    end
    bᵥ = Expr(:(=), :Bᵥ, bᵥ)
    sort!(rinds, rev=true)
    sort!(nrinds, rev=true)
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
    # Push to before reduction loop
    pre = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
    push!(rblock.args, pre)
    # Reduction loop
    for i = 1:length(rinds)
        newblock = Expr(:block)
        push!(block.args,
              Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                              Expr(:call, :axes, :A, rinds[i])), newblock))
        block = newblock
    end
    # Push to inside innermost loop
    reduction = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ,
                                    Expr(:call, Symbol(F.instance), a)))
    push!(block.args, reduction)
    # Push to after reduction loop
    post = Expr(:(=), bᵥ′, :ξ)
    push!(rblock.args, post)
    return quote
        $bᵥ
        @tturbo check_empty=true $loops
        return B
    end
end

function _lvmapreduce3(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        B = hvncat(ntuple(_ -> 1, Val(N)), true, lvmapreduce1(f, op, A))
    else
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), Val(N))
        B = similar(A, Base.promote_op(op, Base.promote_op(f, T)), Dᴮ′)
        _lvmapreduce3!(f, op, init, B, A, Dᴮ′)
    end
    return B
end

@generated function _lvmapreduce3!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, T, N, D}
    mapreduce_quote3(F, OP, I, N, D)
end

# Handle scalar dims by wrapping in Tuple
_lvmapreduce3(f, op, A, dims::Int) = _lvmapreduce3(f, op, A, (dims,))
# Convenience dispatches to match JuliaBase
lvmapreduce3(f, op, A) = lvmapreduce1(f, op, A)
_lvmapreduce3(f, op, A, ::Colon) = lvmapreduce1(f, op, A)

# Common interface for everything related to mapreduce
function lvmapreduce3(f::F, op::OP, init::I, A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {F, OP, I, T, N}
    if (multithreaded === :auto && length(A) > 4095) || multithreaded === true
        # B = init === nothing ? _lvtmapreduce(op, A, dims) : _lvtmapreduce_init(op, A, dims, init)
        B = _lvmapreduce3(f, op, init, A, dims)
    else
        # B = init === nothing ? _lvmapreduce(op, A, dims) : _lvmapreduce_init(op, A, dims, init)
        B = _lvmapreduce3(f, op, init, A, dims)
    end
    return B
end

# Convenience definitions
lvsum3(f, A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N} =
    lvmapreduce3(f, +, zero, A, dims=dims, multithreaded=multithreaded)
lvprod3(f, A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N} =
    lvmapreduce3(f, *, one, A, dims=dims, multithreaded=multithreaded)
lvmaximum3(f, A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N} =
    lvmapreduce3(f, max, typemin, A, dims=dims, multithreaded=multithreaded)
lvminimum3(f, A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N} =
    lvmapreduce3(f, min, typemax, A, dims=dims, multithreaded=multithreaded)
lvextrema3(f, A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N} =
    collect(zip(lvminimum3(f, A, dims=dims, multithreaded=multithreaded),
                lvmaximum3(f, A, dims=dims, multithreaded=multithreaded)))
