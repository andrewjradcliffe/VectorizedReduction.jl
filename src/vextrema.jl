#
# Date created: 2022-06-10
# Author: aradclif
#
#
############################################################################################

using LoopVectorization, Static
_dim(::Type{StaticInt{N}}) where {N} = N::Int

function vextrema(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, Iₘᵢₙ, Iₘₐₓ, T, N, M}
    Dᴬ = size(A)
    Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    Tₒ = Base.promote_op(f, T)
    B = similar(A, Tₒ, Dᴮ)
    C = similar(A, Tₒ, Dᴮ)
    _vextrema!(f, initmin, initmax, B, C, A, dims)
    return collect(zip(B, C))
end
A = rand(3,3,3,3);

vextrema(identity, typemax, typemin, A, (2,4))
extrema(A, dims=(2,4))
vextrema(identity, typemax, typemin, A, (1,2,3,4))
extrema(A, dims=(1,2,3,4))

staticdim_extrema_quote(typeof(typemax), typeof(typemin), [2,4], 4)
branches_extrema_quote(typeof(typemax), typeof(typemin), 4, 2, typeof((2, 4)))

staticdim_extrema_quote(typeof(typemax), typeof(typemin), [1,2,3,4], 4)

using BenchmarkTools, VectorizedReduction

A = rand(10,10,10,10);

@benchmark vextrema(identity, typemax, typemin, A, (2,4))
@benchmark vvextrema(A, (2,4))
@benchmark vextrema(abs2, typemax, typemin, A, (2,4))
@benchmark vvextrema(abs2, A, (2,4))

B = similar(A, (size(A, 1), 1, size(A, 3), 1));
C = similar(A, (size(A, 1), 1, size(A, 3), 1));
@benchmark collect(zip($B, $C))

function staticdim_extrema_quote(Iₘᵢₙ, Iₘₐₓ, static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    Cᵥ = Expr(:call, :view, :C)
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
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B, C), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B, C), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        mn = Expr(:(=), :mn, Expr(:call, Symbol(Iₘᵢₙ.instance), Expr(:call, :eltype, :Bᵥ)))
        mx = Expr(:(=), :mx, Expr(:call, Symbol(Iₘₐₓ.instance), Expr(:call, :eltype, :Cᵥ)))
        push!(rblock.args, mn, mx)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        setb = Expr(:(=), Bᵥ′, :mn)
        setc = Expr(:(=), Cᵥ′, :mx)
        push!(rblock.args, setb, setc)
        return quote
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            @turbo $loops
            return B, C
        end
    else
        # Pre-reduction
        mn = Expr(:(=), :mn, Expr(:call, Symbol(Iₘᵢₙ.instance), Expr(:call, :eltype, :Bᵥ)))
        mx = Expr(:(=), :mx, Expr(:call, Symbol(Iₘₐₓ.instance), Expr(:call, :eltype, :Cᵥ)))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        return quote
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            $mn
            $mx
            @turbo $loops
            Bᵥ[] = mn
            Cᵥ[] = mx
            return B, C
        end
    end
end


function branches_extrema_quote(Iₘᵢₙ, Iₘₐₓ, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vextrema!(f, initmin, initmax, B, C, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vextrema!(f, initmin, initmax, B, C, A, $tc))))
            return q
        end
    end
    return staticdim_extrema_quote(Iₘᵢₙ, Iₘₐₓ, static_dims, N)
end


@generated function _vextrema!(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, Iₘᵢₙ, Iₘₐₓ, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_extrema_quote(Iₘᵢₙ, Iₘₐₓ, N, M, D)
end

################

B = similar(A, Tuple{eltype(A), eltype(A)}, (size(A, 1), 1, size(A, 3), 1))
static_dims = [2,4]
N = 4
Iₘᵢₙ, Iₘₐₓ = typeof(typemax), typeof(typemin)

A = rand(3,3,3,3);
A = rand(10,10,10,10);

vextrema_nonzip(identity, typemax, typemin, A, (2,4))
extrema(A, dims=(2,4))
vextrema_nonzip(identity, typemax, typemin, A, (1,2,3,4))
extrema(A, dims=(1,2,3,4))

staticdim_extrema_nonzip_quote(typeof(typemax), typeof(typemin), [2,4], 4)
branches_extrema_nonzip_quote(typeof(typemax), typeof(typemin), 4, 2, typeof((2, 4)))
staticdim_extrema_nonzip_quote(typeof(typemax), typeof(typemin), [1,2,3,4], 4)

@benchmark vextrema(identity, typemax, typemin, $A, $(2,4))
@benchmark vextrema_nonzip(identity, typemax, typemin, $A, $(2,4))
@benchmark vvextrema($A, $(2,4))
@benchmark extrema($A, dims=$(2,4))
@benchmark vextrema(abs2, typemax, typemin, $A, $(2,4))
@benchmark vextrema_nonzip(abs2, typemax, typemin, $A, $(2,4))
@benchmark vvextrema(abs2, $A, $(2,4))
@benchmark extrema(abs2, $A, dims=$(2,4))

B = similar(A, Tuple{eltype(A), eltype(A)}, (size(A, 1), 1, size(A, 3), 1));

@code_warntype vextrema_nonzip(identity, typemax, typemin, A, (2,4))
@code_warntype _vextrema_nonzip!(identity, typemax, typemin, B, A, (2,4))
@benchmark _vextrema_nonzip!($identity, $typemax, $typemin, $B, $A, $(2,4))

@code_warntype _vextrema_nonzip!(identity, typemax, typemin, B, A, (static(2),static(4)))


function vextrema_nonzip(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, Iₘᵢₙ, Iₘₐₓ, T, N, M}
    Dᴬ = size(A)
    Dᴮ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    Tₒ = Base.promote_op(f, T)
    B = similar(A, Tuple{Tₒ, Tₒ}, Dᴮ)
    _vextrema_nonzip!(f, initmin, initmax, B, A, dims)
    return B
end

function staticdim_extrema_nonzip_quote(Iₘᵢₙ, Iₘₐₓ, static_dims::Vector{Int}, N::Int)
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
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        mn = Expr(:(=), :mn, Expr(:call, Symbol(Iₘᵢₙ.instance), Expr(:call, :eltype, Expr(:call, :eltype, :Bᵥ))))
        mx = Expr(:(=), :mx, Expr(:call, Symbol(Iₘₐₓ.instance), Expr(:call, :eltype, Expr(:call, :eltype, :Bᵥ))))
        push!(rblock.args, mn, mx)
        # # Reduction loop
        # for d ∈ rinds
        #     newblock = Expr(:block)
        #     push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
        #     block = newblock
        # end
        # Reduction loop alt
        newblock = Expr(:block)
        d = first(rinds)
        il1 = Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock)
        push!(block.args, :(@turbo $il1))
        # push!(block.args, quote @turbo $il1 end)
        # push!(block.args, Expr(:macrocall, Symbol("@turbo"), il1))
        block = newblock
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # q1 = quote @turbo $il1 end
        # q2 = :(@turbo $il1)
        # q3 = Expr(:macrocall, Symbol("@turbo"), il1)
        # Meta.show_sexpr(q1)
        # Meta.show_sexpr(q2)
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        setb = Expr(:(=), Bᵥ′, Expr(:tuple, :mn, :mx))
        push!(rblock.args, setb)
        return quote
            Bᵥ = $Bᵥ
            $loops
            return B
        end
    else
        # Pre-reduction
        mn = Expr(:(=), :mn, Expr(:call, Symbol(Iₘᵢₙ.instance), Expr(:call, :eltype, Expr(:call, :eltype, :Bᵥ))))
        mx = Expr(:(=), :mx, Expr(:call, Symbol(Iₘₐₓ.instance), Expr(:call, :eltype, Expr(:call, :eltype, :Bᵥ))))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setmin = Expr(:(=), :mn, Expr(:call, :min, Expr(:call, :f, A), :mn))
        setmax = Expr(:(=), :mx, Expr(:call, :max, Expr(:call, :f, A), :mx))
        push!(block.args, setmin, setmax)
        return quote
            Bᵥ = $Bᵥ
            $mn
            $mx
            @turbo $loops
            Bᵥ[] = (mn, mx)
            return B
        end
    end
end

function branches_extrema_nonzip_quote(Iₘᵢₙ, Iₘₐₓ, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vextrema_nonzip!(f, initmin, initmax, B, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vextrema_nonzip!(f, initmin, initmax, B, A, $tc))))
            return q
        end
    end
    return staticdim_extrema_nonzip_quote(Iₘᵢₙ, Iₘₐₓ, static_dims, N)
end


@generated function _vextrema_nonzip!(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, B::AbstractArray{Tuple{Tₒ, Tₒ}, N}, A::AbstractArray{T, N}, dims::D) where {F, Iₘᵢₙ, Iₘₐₓ, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_extrema_nonzip_quote(Iₘᵢₙ, Iₘₐₓ, N, M, D)
end
