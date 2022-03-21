#
# Date created: 2022-03-21
# Author: aradclif
#
#
############################################################################################
# Attempts at findmax, argmax

function findmax_quote(N::Int, D)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Cᵥ = Expr(:call, :view, :C)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    Cᵥ′ = Expr(:ref, :Cᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if D.parameters[d] === Static.One
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
    Bᵥ = Expr(:(=), :Bᵥ, Bᵥ)
    Cᵥ = Expr(:(=), :Cᵥ, Cᵥ)
    reverse!(rinds)
    reverse!(nrinds)
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), Symbol(:i_, nrinds[1]),
                            Expr(:call, :indices, Expr(:tuple, :A, :B, :C), nrinds[1])), block)
    for i = 2:length(nrinds)
        newblock = Expr(:block)
        push!(block.args,
              Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
                              Expr(:call, :indices, Expr(:tuple, :A, :B, :C), nrinds[i])), newblock))
        block = newblock
    end
    rblock = block
    # Pre-reduction
    v = Expr(:(=), :v, Expr(:call, :typemin, Expr(:call, :eltype, :Bᵥ)))
    push!(rblock.args, v)
    for d ∈ rinds
        j = Expr(:(=), Symbol(:j_, d), Expr(:call, :one, :Int))
        push!(rblock.args, j)
    end
    # Reduction loop
    for i = 1:length(rinds)
        newblock = Expr(:block)
        push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                                          Expr(:call, :axes, :A, rinds[i])), newblock))
        block = newblock
    end
    # Push to inside innermost loop
    cmpr = Expr(:(=), :newmax, Expr(:call, :(>), a, :v))
    push!(block.args, cmpr)
    setmax = Expr(:(=), :v, Expr(:call, :ifelse, :newmax, a, :v))
    push!(block.args, setmax)
    for d ∈ rinds
        setj = Expr(:(=), Symbol(:j_, d), Expr(:call, :ifelse, :newmax, Symbol(:i_, d), Symbol(:j_, d)))
        push!(block.args, setj)
    end
    # Push to after reduction loop
    postmax = Expr(:(=), Bᵥ′, :v)
    push!(rblock.args, postmax)
    # # Simplest variety
    # postj = Expr(:(=), Cᵥ′, partialtermrawj(Tuple(rinds)))
    # push!(rblock.args, postj)
    # Potential loop-carried dependency
    jterm = partialtermrawj(Tuple(rinds))
    for d ∈ nrinds
        push!(jterm.args, singletermraw(d))
    end
    push!(jterm.args, 1)
    push!(jterm.args, :Dstar)
    postj = Expr(:(=), Cᵥ′, jterm)
    push!(rblock.args, postj)
    # sz = sizeblock(N)
    t = Expr(:tuple)
    for d = 1:N
        push!(t.args, Symbol(:D_, d))
    end
    sz = Expr(:(=), t, Expr(:call, :size, :A))
    dstar = totaloffsetraw(N)
    dstar′ = Expr(:(=), :Dstar, Expr(:call, :-, :Dstar))
    return quote
        $sz
        $dstar
        $dstar′
        $Bᵥ
        $Cᵥ
        @turbo $loops
    end
end

function vfindmax(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), Val(N))
    B = similar(A, Dᴮ′)
    C = similar(A, Int, Dᴮ′)
    _vfindmax!(B, C, A, Dᴮ′)
    B, C
end

@generated function _vfindmax!(B::AbstractArray{T, N}, C::AbstractArray{Int, N},
                               A::AbstractArray{T, N}, dims::D) where {T, N, D}
    findmax_quote(N, D)
end

B, C = vfindmax(A, dims)
findmax_quote(N, D)
CartesianIndices(A)[C] == argmax(A, dims=dims)
@benchmark vfindmax(A, dims)
@benchmark findmax(A, dims=dims)
