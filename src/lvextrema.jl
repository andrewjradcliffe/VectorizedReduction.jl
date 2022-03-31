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
        return B, C
    end
end

function vextrema2(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    Dᴮ′ = ntuple(d -> d ∈ dims ? StaticInt(1) : size(A, d), Val(N))
    B = similar(A, Dᴮ′)
    C = similar(A, Dᴮ′)
    _vextrema!(B, C, A, Dᴮ′)
    B, C
    collect(zip(C, B))
end

@generated function _vextrema!(B::AbstractArray{T, N}, C::AbstractArray{T, N},
                               A::AbstractArray{T, N}, dims::D) where {T, N, D}
    extrema_quote(N, D)
end

################
# Compile-time determination rather than using dispatch system
_dim(::Type{StaticInt{N}}) where {N} = N::Int
# Stable return type version, otherwise it's just a suggestion
# _dim2(::Type{StaticInt{N}})::Int where {N} = N

function vvvextrema(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Dᴮ′)
    C = similar(A, Dᴮ′)
    _vvvextrema!(B, C, A, dims)
    collect(zip(C, B))
end

function staticdim_extrema_quote(static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Cᵥ = Expr(:call, :view, :C)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    Cᵥ′ = Expr(:ref, :Cᵥ)
    rinds = Int[]
    nrinds = Int[]
    for d = 1:N
        if d ∈ static_dims
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
    reverse!(rinds)
    reverse!(nrinds)
    if !isempty(nrinds)
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
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            @turbo $loops
            return B, C
        end
    else
        # Pre-reduction
        vmax = Expr(:(=), :vmax, Expr(:call, :typemin, Expr(:call, :eltype, :Bᵥ)))
        vmin = Expr(:(=), :vmin, Expr(:call, :typemax, Expr(:call, :eltype, :Cᵥ)))
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
        newmax = Expr(:(=), :newmax, Expr(:call, :(>), A, :vmax))
        newmin = Expr(:(=), :newmin, Expr(:call, :(<), A, :vmin))
        setvmax = Expr(:(=), :vmax, Expr(:call, :ifelse, :newmax, A, :vmax))
        setvmin = Expr(:(=), :vmin, Expr(:call, :ifelse, :newmin, A, :vmin))
        push!(block.args, newmax, newmin, setvmax, setvmin)
        return quote
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            $vmax
            $vmin
            @turbo $loops
            Bᵥ[] = vmax
            Cᵥ[] = vmin
            return B, C
        end
    end
end

function branches_extrema_quote(N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vvvextrema!(B, C, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vvvextrema!(B, C, A, $tc))))
            return q
        end
    end
    return staticdim_extrema_quote(static_dims, N)
end

@generated function _vvvextrema!(B::AbstractArray{T, N}, C::AbstractArray{T, N},
                                A::AbstractArray{T, N}, dims::D) where {T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_extrema_quote(N, M, D)
end
@generated function _vvvextrema!(B::AbstractArray{T, N}, C::AbstractArray{T, N},
                                A::AbstractArray{T, N}, dims::Tuple{}) where {T, N}
    :(copyto!(B, A); copyto!(C, A); return B, C)
end

