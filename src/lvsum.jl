#
# Date created: 2022-03-20
# Author: aradclif
#
#
############################################################################################
# To eliminate two allocations, define separately, rather than using reduce
function sum_quote3(N::Int, D)
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
    pre = Expr(:(=), :ξ, Expr(:call, :zero, Expr(:call, :eltype, :Bᵥ)))
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
    reduction = Expr(:(=), :ξ, Expr(:call, :+, :ξ, a))
    # reduction = :(ξ += $a)
    push!(block.args, reduction)
    # Push to after reduction loop
    post = Expr(:(=), bᵥ′, :ξ)
    push!(rblock.args, post)
    return quote
        $bᵥ
        @tturbo $loops
        return B
    end
end

function _lvsum3(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        B = hvncat(ntuple(_ -> 1, Val(N)), true, _lvsum1(A))
    else
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), Val(N))
        B = similar(A, Base.promote_op(+, T, Int), Dᴮ′)
        _lvsum3!(B, A, Dᴮ′)
    end
    return B
end
@generated function _lvsum3!(B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {Tₒ, T, N, D}
    sum_quote3(N, D)
end
function _lvsum1(A::AbstractArray{T, N}) where {T, N}
    ξ = zero(Base.promote_op(+, T, Int))
    @turbo for i ∈ eachindex(A)
        ξ += A[i]
    end
    ξ
end

# Handle scalar dims by wrapping in Tuple
_lvsum3(A, dims::Int) = _lvsum3(A, (dims,))
# Convenience dispatches to match JuliaBase
lvsum3(A) = _lvsum1(A)
_lvsum3(A, ::Colon) = _lvsum1(A)

# Common interface for everything related to reduce
function lvsum4(A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N}
    if (multithreaded === :auto && length(A) > 4095) || multithreaded === true
        # B = init === nothing ? _lvtsum(op, A, dims) : _lvtsum_init(op, A, dims, init)
        B = _lvsum3(A, dims)
    else
        # B = init === nothing ? _lvsum(op, A, dims) : _lvsum_init(op, A, dims, init)
        B = _lvsum3(A, dims)
    end
    return B
end

function lvsum3(A::AbstractArray{T, N}; dims=:, multithreaded=:auto) where {T, N}
    if (multithreaded === :auto && length(A) > 4095) || multithreaded === true
        # B = init === nothing ? _lvtsum(op, A, dims) : _lvtsum_init(op, A, dims, init)
        B = _lvreduce3(+, zero, A, dims)
    else
        # B = init === nothing ? _lvsum(op, A, dims) : _lvsum_init(op, A, dims, init)
        B = _lvreduce3(+, zero, A, dims)
    end
    return B
end

################

