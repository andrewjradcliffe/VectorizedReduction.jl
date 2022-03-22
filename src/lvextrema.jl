#
# Date created: 2022-03-22
# Author: aradclif
#
#
############################################################################################
# Combined maximum, minimum

function extrema_quote(N::Int, D)
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
    vmax = Expr(:(=), :vmax, Expr(:call, :typemin, Expr(:call, :eltype, :Bᵥ)))
    vmin = Expr(:(=), :vmin, Expr(:call, :typemax, Expr(:call, :eltype, :Cᵥ)))
    push!(rblock.args, vmax, vmin)
    # Reduction loop
    for d ∈ rinds
        newblock = Expr(:block)
        push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :axes, :A, d)), newblock))
        block = newblock
    end
    newmax = Expr(:(=), :newmax, Expr(:call, :(>), A, :vmax))
    newmin = Expr(:(=), :newmin, Expr(:call, :(<), A, :vmin))
    setvmax = Expr(:(=), :vmax, Expr(:call, :ifelse, :newmax, A, :vmax))
    setvmin = Expr(:(=), :vmin, Expr(:call, :ifelse, :newmin, A, :vmin))
    push!(block.args, newmax, newmin, setvmax, setvmin)
    setb = Expr(:(=), Bᵥ′, :vmax)
    setc = Expr(:(=), Cᵥ′, :vmin)
    push!(rblock.args, setb, setc)
    return quote
        $Bᵥ
        $Cᵥ
        @turbo $loops
    end
end

function vextrema2(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), Val(N))
    B = similar(A, Dᴮ′)
    C = similar(A, Dᴮ′)
    _vextrema!(B, C, A, Dᴮ′)
    B, C
end

@generated function _vextrema!(B::AbstractArray{T, N}, C::AbstractArray{T, N},
                               A::AbstractArray{T, N}, dims::D) where {T, N, D}
    extrema_quote(N, D)
end

@benchmark vextrema2(A, dims)
@benchmark extrema(A, dims=dims)
@benchmark vextrema(A, dims=dims)


