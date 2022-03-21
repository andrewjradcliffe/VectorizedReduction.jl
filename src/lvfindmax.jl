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
    cmpr = Expr(:(=), :newmax, Expr(:call, :(>), A, :v))
    push!(block.args, cmpr)
    setmax = Expr(:(=), :v, Expr(:call, :ifelse, :newmax, A, :v))
    push!(block.args, setmax)
    for d ∈ rinds
        setj = Expr(:(=), Symbol(:j_, d), Expr(:call, :ifelse, :newmax, Symbol(:i_, d), Symbol(:j_, d)))
        push!(block.args, setj)
    end
    # Push to after reduction loop
    postmax = Expr(:(=), Bᵥ′, :v)
    push!(rblock.args, postmax)
    # # Simplest variety
    # postj = Expr(:(=), Cᵥ′, length(rinds) == 1 ? singletermrawj(first(rinds)) : partialtermrawj(Tuple(rinds)))
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
        return B, C
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

A = rand(10,10);
dims=(1,)
N = ndims(A)
Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), N)
D = typeof(Dᴮ′)
B, C = vfindmax(A, dims);
findmax_quote(N, D)
CartesianIndices(A)[C] == argmax(A, dims=dims)
@benchmark vfindmax(A, dims)
@benchmark findmax(A, dims=dims)
@benchmark bfindmax(A, dims)
bfindmax_quote(N, D)


B = similar(A, Dᴮ′)
C = similar(A, Int, Dᴮ′)

ls0 = :(for i_1 = indices((A, B, C), 1)
           v = typemin(eltype(Bᵥ))
           j_2 = one(Int)
           for i_2 = axes(A, 2)
               newmax = A[i_1, i_2] > v
               v = ifelse(newmax, A[i_1, i_2], v)
               j_2 = ifelse(newmax, i_2, j_2)
           end
           Bᵥ[i_1] = v
           Cᵥ[i_1] = D_1 * j_2 + i_1 + 1 + Dstar
       end);
ls = LoopVectorization.LoopSet(ls0);
ops = LoopVectorization.operations(ls)
deps = LoopVectorization.loopdependencies.(ops);

ls02 = :(for i_2 = indices((A, B, C), 2)
             v = typemin(eltype(Bᵥ))
             j_1 = one(Int)
             for i_1 = axes(A, 1)
                 newmax = A[i_1, i_2] > v
                 v = ifelse(newmax, A[i_1, i_2], v)
                 j_1 = ifelse(newmax, i_1, j_1)
             end
             Bᵥ[i_2] = v
             Cᵥ[i_2] = j_1 + D_1 * i_2 + 1 + Dstar
         end);
ls2 = LoopVectorization.LoopSet(ls02);
ops2 = LoopVectorization.operations(ls2)

# following sequence
q = @macroexpand @turbo for i_2 = indices((A, B, C), 2)
    v = typemin(eltype(Bᵥ))
    j_1 = one(Int)
    for i_1 = axes(A, 1)
        newmax = A[i_1, i_2] > v
        v = ifelse(newmax, A[i_1, i_2], v)
        j_1 = ifelse(newmax, i_1, j_1)
    end
    Bᵥ[i_2] = v
    Cᵥ[i_2] = j_1 + D_1 * i_2 + 1 + Dstar
end;

LoopVectorization.avx_body(ls2, )

function foo!(B, C, A)
    (D_1, D_2) = size(A)
    Dstar = (*)(1) + (*)(D_1)
    Dstar = -Dstar
    Bᵥ = view(B, firstindex(B, 1), Colon())
    Cᵥ = view(C, firstindex(C, 1), Colon())
    @macroexpand @turbo for i_2 = indices((A, B, C), 2)
        v = typemin(eltype(Bᵥ))
        j_1 = one(Int)
        for i_1 = axes(A, 1)
            newmax = A[i_1, i_2] > v
            v = ifelse(newmax, A[i_1, i_2], v)
            j_1 = ifelse(newmax, i_1, j_1)
        end
        Bᵥ[i_2] = v
        Cᵥ[i_2] = j_1 + D_1 * i_2 + 1 + Dstar
    end
    B, C
end

@enter vfindmax(A, dims)

@enter foo!(B, C, A)
