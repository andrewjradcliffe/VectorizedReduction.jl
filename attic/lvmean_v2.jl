#
# Date created: 2022-03-20
# Author: aradclif
#
#
############################################################################################
function mean_quote3(N::Int, D)
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    bᵥ = Expr(:call, :view, :B)
    bᵥ′ = Expr(:ref, :Bᵥ)
    denom = Expr(:call, :*)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if D.parameters[d] === Static.One
            push!(bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(rinds, d)
            push!(denom.args, Expr(:call, :size, :A, d))
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
    push!(block.args, reduction)
    # Push to after reduction loop
    post = Expr(:(=), bᵥ′, Expr(:call, :*, :ξ, :invdenom))
    push!(rblock.args, post)
    return quote
        denom = $denom
        invdenom = inv(denom)
        $bᵥ
        @tturbo check_empty=true $loops
        return B
    end
end

function _lvmean3(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        B = hvncat(ntuple(_ -> 1, Val(N)), true, lvmean1(A))
    else
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), Val(N))
        B = similar(A, Base.promote_op(/, T, Int), Dᴮ′)
        _lvmean3!(B, A, Dᴮ′)
    end
    return B
end

@generated function _lvmean3!(B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {Tₒ, T, N, D}
    mean_quote3(N, D)
end

function mean_quote3(F, N::Int, D)
    a = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    bᵥ = Expr(:call, :view, :B)
    bᵥ′ = Expr(:ref, :Bᵥ)
    denom = Expr(:call, :*)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if D.parameters[d] === Static.One
            push!(bᵥ.args, Expr(:call, :firstindex, :B, d))
            push!(rinds, d)
            push!(denom.args, Expr(:call, :size, :A, d))
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
    reduction = Expr(:(=), :ξ, Expr(:call, :+, :ξ, Expr(:call, Symbol(F.instance), a)))
    push!(block.args, reduction)
    # Push to after reduction loop
    post = Expr(:(=), bᵥ′, Expr(:call, :*, :ξ, :invdenom))
    push!(rblock.args, post)
    return quote
        denom = $denom
        invdenom = inv(denom)
        $bᵥ
        @tturbo check_empty=true $loops
        return B
    end
end

function _lvmean3(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    if ntuple(identity, Val(N)) ⊆ dims
        B = hvncat(ntuple(_ -> 1, Val(N)), true, lvmean1(f, A))
    else
        Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), Val(N))
        B = similar(A, Base.promote_op(/, Base.promote_op(f, T), Int), Dᴮ′)
        _lvmean3!(f, B, A, Dᴮ′)
    end
    return B
end

@generated function _lvmean3!(f::F, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, Tₒ, T, N, D}
    mean_quote3(F, N, D)
end
