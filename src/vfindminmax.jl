#
# Date created: 2022-03-24
# Author: aradclif
#
#
############################################################################################
# Version of findmax from lvfindmax.jl, but using branching + work function

# There are some clear motivations for using allowing `f` to be anonymous
# Should separate versions exist, given that there are only two? No -- the generated
# functions are necessary anyway.
# Anonymous `f` in the arbitrary-dimensional case also makes sense, and, moreover,
# the @generated call is going to specialize anyway. The motivation is for
# function calls from within some other compiled code, where the anonymous function
# is itself compiled -- noting that any repetition of anonymous functions
# strongly motivates naming the function so that the branching discussed in
# combinatorial.jl is not repeated unnecessarily.

function vfindminmax(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(f, T), Dᴮ′)
    C = similar(A, Int, Dᴮ′)
    _vfindminmax!(f, op, init, B, C, A, dims)
    return B, CartesianIndices(A)[C]
end

# Convenience defintions
vfindmax(f::F, A, dims) where {F<:Function} = vfindminmax(f, >, typemin, A, dims)
vfindmin(f::F, A, dims) where {F<:Function} = vfindminmax(f, <, typemax, A, dims)
vfindmax(A, dims) = vfindminmax(identity, >, typemin, A, dims)
vfindmin(A, dims) = vfindminmax(identity, <, typemax, A, dims)

# over all dims
@generated function vfindminmax(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I, T, N}
    # fsym = F.instance
    opsym = OP.instance
    initsym = I.instance
    # Tₒ = promote_type(Base.promote_op(fsym, T), T) # attempt at protecting against Union{}
    quote
        # m = $initsym($Tₒ)
        m = $initsym(Base.promote_op(f, $T))
        j = 0
        @turbo for i ∈ eachindex(A)
            newm = $opsym(f(A[i]), m)
            m = ifelse(newm, f(A[i]), m)
            j = ifelse(newm, i, j)
        end
        return m, CartesianIndices(A)[j]
    end
end

# A seemingly superfluous dispatch, but is necessary to enforce that a vector returns
# a linear index.
@generated function vfindminmax(f::F, op::OP, init::I, A::AbstractVector{T}, ::Colon) where {F, OP, I, T}
    opsym = OP.instance
    initsym = I.instance
    quote
        m = $initsym(Base.promote_op(f, $T))
        j = 0
        @turbo for i ∈ eachindex(A)
            newm = $opsym(f(A[i]), m)
            m = ifelse(newm, f(A[i]), m)
            j = ifelse(newm, i, j)
        end
        return m, j
    end
end

vfindmax(f::F, A) where {F<:Function} = vfindminmax(f, >, typemin, A, :)
vfindmin(f::F, A) where {F<:Function} = vfindminmax(f, <, typemax, A, :)
vfindmax(A) = vfindmax(identity, A)
vfindmin(A) = vfindmin(identity, A)

function staticdim_findminmax_quote(OP, I, static_dims::Vector{Int}, N::Int)
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
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B, C), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B, C), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        push!(rblock.args, ξ)
        for d ∈ rinds
            push!(rblock.args, Expr(:(=), Symbol(:j_, d), Expr(:call, :one, :Int)))
        end
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        # cmpr = Expr(:(=), :newmax, Expr(:call, :(>), A, :ξ))
        cmpr = Expr(:(=), :newmax, Expr(:call, Symbol(OP.instance),
                                        Expr(:call, :f, A), :ξ))
        push!(block.args, cmpr)
        # setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax, A, :ξ))
        setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax,
                                     Expr(:call, :f, A), :ξ))
        push!(block.args, setmax)
        for d ∈ rinds
            setj = Expr(:(=), Symbol(:j_, d),
                        Expr(:call, :ifelse, :newmax, Symbol(:i_, d), Symbol(:j_, d)))
            push!(block.args, setj)
        end
        # Push to after reduction loop
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        # # Simplest variety
        # postj = Expr(:(=), Cᵥ′)
        # if length(rinds) == 1
        #     push!(postj.args, d == 1 ? :j_1 :
        #         Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
        # else
        #     setc = Expr(:call, :+)
        #     for d ∈ rinds
        #         push!(setc.args, d == 1 ? :j_1 :
        #             Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
        #     end
        #     push!(postj.args, setc)
        # end
        # Potential loop-carried dependency
        setc = Expr(:call, :+)
        for d ∈ rinds
            push!(setc.args, d == 1 ? :j_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
        end
        for d ∈ nrinds
            push!(setc.args, d == 1 ? :i_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:i_, d)))
        end
        push!(setc.args, 1, :Dstar)
        postj = Expr(:(=), Cᵥ′, setc)
        push!(rblock.args, postj)
        # strides, offsets
        t = Expr(:tuple)
        for d = 1:N
            push!(t.args, Symbol(:D_, d))
        end
        sz = Expr(:(=), t, Expr(:call, :size, :A))
        dstar = Expr(:call, :+, 1)
        for d = 2:N
            push!(dstar.args, d == 2 ? :D_1 : Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
        end
        return quote
            $sz
            Dstar = $dstar
            Dstar = -Dstar
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            @turbo $loops
            return B, C
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        j = Expr(:tuple)
        for d = 1:N
            push!(j.args, Symbol(:j_, d))
        end
        js = :($j = $(ntuple(_ -> 1, Val(N))))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        # cmpr = Expr(:(=), :newmax, Expr(:call, :(>), A, :ξ))
        cmpr = Expr(:(=), :newmax, Expr(:call, Symbol(OP.instance),
                                        Expr(:call, :f, A), :ξ))
        push!(block.args, cmpr)
        # setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax, A, :ξ))
        setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax,
                                     Expr(:call, :f, A), :ξ))
        push!(block.args, setmax)
        for d ∈ rinds
            setj = Expr(:(=), Symbol(:j_, d),
                        Expr(:call, :ifelse, :newmax, Symbol(:i_, d), Symbol(:j_, d)))
            push!(block.args, setj)
        end
        setc = Expr(:call, :+)
        for d ∈ rinds
            push!(setc.args, d == 1 ? :j_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
        end
        for d ∈ nrinds
            push!(setc.args, d == 1 ? :i_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:i_, d)))
        end
        push!(setc.args, 1, :Dstar)
        # strides, offsets
        t = Expr(:tuple)
        for d = 1:N
            push!(t.args, Symbol(:D_, d))
        end
        sz = Expr(:(=), t, Expr(:call, :size, :A))
        dstar = Expr(:call, :+, 1)
        for d = 2:N
            push!(dstar.args, d == 2 ? :D_1 : Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
        end
        return quote
            $js
            $sz
            Dstar = $dstar
            Dstar = -Dstar
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            $ξ
            @turbo $loops
            Bᵥ[] = ξ
            Cᵥ[] = $setc
            return B, C
        end
    end
end

function branches_findminmax_quote(OP, I, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vfindminmax!(f, op, init, B, C, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vfindminmax!(f, op, init, B, C, A, $tc))))
            return q
        end
    end
    return staticdim_findminmax_quote(OP, I, static_dims, N)
end

@generated function _vfindminmax!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₗ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, Tₗ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_findminmax_quote(OP, I, N, M, D)
end
@generated function _vfindminmax!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₗ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, Tₗ, T, N}
    :(copyto!(B, A); copyto!(C, LinearIndices(A)); return B, C)
end


################
# function vfindminmax2(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
#     Dᴬ = size(A)
#     Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
#     B = similar(A, Base.promote_op(f, T), Dᴮ′)
#     C = similar(A, Int, Dᴮ′)
#     _vfindminmax2!(f, op, init, B, C, A, dims)
#     return B, CartesianIndices(A)[C]
# end

# function staticdim_findminmax2_quote(OP, I, static_dims::Vector{Int}, N::Int)
#     A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
#     Bᵥ = Expr(:call, :view, :B)
#     Cᵥ = Expr(:call, :view, :C)
#     Bᵥ′ = Expr(:ref, :Bᵥ)
#     Cᵥ′ = Expr(:ref, :Cᵥ)
#     rinds = Int[]
#     nrinds = Int[]
#     for d = 1:N
#         if d ∈ static_dims
#             push!(Bᵥ.args, Expr(:call, :firstindex, :B, d))
#             push!(Cᵥ.args, Expr(:call, :firstindex, :C, d))
#             push!(rinds, d)
#         else
#             push!(Bᵥ.args, :)
#             push!(Cᵥ.args, :)
#             push!(nrinds, d)
#             push!(Bᵥ′.args, Symbol(:i_, d))
#             push!(Cᵥ′.args, Symbol(:i_, d))
#         end
#     end
#     reverse!(rinds)
#     reverse!(nrinds)
#     if !isempty(nrinds)
#         block = Expr(:block)
#         loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B, C), $(nrinds[1]))), block)
#         for d ∈ @view(nrinds[2:end])
#             newblock = Expr(:block)
#             push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B, C), $d)), newblock))
#             block = newblock
#         end
#         rblock = block
#         # Pre-reduction
#         ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
#         push!(rblock.args, ξ)
#         for d ∈ rinds
#             push!(rblock.args, Expr(:(=), Symbol(:j_, d), Expr(:call, :one, :Int)))
#         end
#         # Reduction loop
#         for d ∈ rinds
#             newblock = Expr(:block)
#             push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
#             block = newblock
#         end
#         # Push to inside innermost loop
#         # cmpr = Expr(:(=), :newmax, Expr(:call, :(>), A, :ξ))
#         cmpr = Expr(:(=), :newmax, Expr(:call, Symbol(OP.instance),
#                                         Expr(:call, :f, A), :ξ))
#         push!(block.args, cmpr)
#         # setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax, A, :ξ))
#         setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax,
#                                      Expr(:call, :f, A), :ξ))
#         push!(block.args, setmax)
#         for d ∈ rinds
#             setj = Expr(:(=), Symbol(:j_, d),
#                         Expr(:call, :ifelse, :newmax, Symbol(:i_, d), Symbol(:j_, d)))
#             push!(block.args, setj)
#         end
#         # Push to after reduction loop
#         setb = Expr(:(=), Bᵥ′, :ξ)
#         push!(rblock.args, setb)
#         # # Simplest variety
#         # postj = Expr(:(=), Cᵥ′)
#         # if length(rinds) == 1
#         #     push!(postj.args, d == 1 ? :j_1 :
#         #         Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
#         # else
#         #     setc = Expr(:call, :+)
#         #     for d ∈ rinds
#         #         push!(setc.args, d == 1 ? :j_1 :
#         #             Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
#         #     end
#         #     push!(postj.args, setc)
#         # end
#         # Potential loop-carried dependency
#         setc = Expr(:call, :+)
#         for d ∈ rinds
#             push!(setc.args, d == 1 ? :j_1 :
#                 Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
#         end
#         for d ∈ nrinds
#             push!(setc.args, d == 1 ? :i_1 :
#                 Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:i_, d)))
#         end
#         push!(setc.args, 1, :Dstar)
#         postj = Expr(:(=), Cᵥ′, setc)
#         push!(rblock.args, postj)
#         # strides, offsets
#         t = Expr(:tuple)
#         for d = 1:N
#             push!(t.args, Symbol(:D_, d))
#         end
#         sz = Expr(:(=), t, Expr(:call, :size, :A))
#         dstar = Expr(:call, :+, 1)
#         for d = 2:N
#             push!(dstar.args, d == 2 ? :D_1 : Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
#         end
#         return quote
#             $sz
#             Dstar = $dstar
#             Dstar = -Dstar
#             Bᵥ = $Bᵥ
#             Cᵥ = $Cᵥ
#             @turbo $loops
#             return B, C
#         end
#     else
#         # Pre-reduction
#         ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
#         j = Expr(:tuple)
#         for d = 1:N
#             push!(j.args, Symbol(:j_, d))
#         end
#         js = :($j = $(ntuple(_ -> 1, Val(N))))
#         # Reduction loop
#         block = Expr(:block)
#         loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
#         for d ∈ @view(rinds[2:end])
#             newblock = Expr(:block)
#             push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
#             block = newblock
#         end
#         # Push to inside innermost loop
#         # cmpr = Expr(:(=), :newmax, Expr(:call, :(>), A, :ξ))
#         cmpr = Expr(:(=), :newmax, Expr(:call, Symbol(OP.instance),
#                                         Expr(:call, :f, A), :ξ))
#         push!(block.args, cmpr)
#         # setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax, A, :ξ))
#         setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newmax,
#                                      Expr(:call, :f, A), :ξ))
#         push!(block.args, setmax)
#         for d ∈ rinds
#             setj = Expr(:(=), Symbol(:j_, d),
#                         Expr(:call, :ifelse, :newmax, Symbol(:i_, d), Symbol(:j_, d)))
#             push!(block.args, setj)
#         end
#         setc = Expr(:call, :+)
#         for d ∈ rinds
#             push!(setc.args, d == 1 ? :j_1 :
#                 Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
#         end
#         for d ∈ nrinds
#             push!(setc.args, d == 1 ? :i_1 :
#                 Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:i_, d)))
#         end
#         push!(setc.args, 1, :Dstar)
#         # strides, offsets
#         t = Expr(:tuple)
#         for d = 1:N
#             push!(t.args, Symbol(:D_, d))
#         end
#         sz = Expr(:(=), t, Expr(:call, :size, :A))
#         dstar = Expr(:call, :+, 1)
#         for d = 2:N
#             push!(dstar.args, d == 2 ? :D_1 : Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
#         end
#         return quote
#             $js
#             $sz
#             Dstar = $dstar
#             Dstar = -Dstar
#             Bᵥ = $Bᵥ
#             Cᵥ = $Cᵥ
#             $ξ
#             @turbo $loops
#             Bᵥ[] = ξ
#             Cᵥ[] = $setc
#             return B, C
#         end
#     end
# end

# function branches_findminmax2_quote(OP, I, N::Int, M::Int, D)
#     static_dims = Int[]
#     for m ∈ 1:M
#         param = D.parameters[m]
#         if param <: StaticInt
#             new_dim = _dim(param)::Int
#             push!(static_dims, new_dim)
#         else
#             # tuple of static dimensions
#             t = Expr(:tuple)
#             for n ∈ static_dims
#                 push!(t.args, :(StaticInt{$n}()))
#             end
#             q = Expr(:block, :(dimm = dims[$m]))
#             qold = q
#             # if-elseif statements
#             ifsym = :if
#             for n ∈ 1:N
#                 n ∈ static_dims && continue
#                 tc = copy(t)
#                 push!(tc.args, :(StaticInt{$n}()))
#                 qnew = Expr(ifsym, :(dimm == $n), :(return _vfindminmax2!(f, op, init, B, C, A, $tc)))
#                 for r ∈ m+1:M
#                     push!(tc.args, :(dims[$r]))
#                 end
#                 push!(qold.args, qnew)
#                 qold = qnew
#                 ifsym = :elseif
#             end
#             # else statement
#             tc = copy(t)
#             for r ∈ m+1:M
#                 push!(tc.args, :(dims[$r]))
#             end
#             push!(qold.args, Expr(:block, :(return _vfindminmax2!(f, op, init, B, C, A, $tc))))
#             return q
#         end
#     end
#     return staticdim_findminmax2_quote(OP, I, static_dims, N)
# end

# @generated function _vfindminmax2!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₗ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, Tₗ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
#     branches_findminmax2_quote(OP, I, N, M, D)
# end
# @generated function _vfindminmax2!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₗ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, Tₗ, T, N}
#     :(copyto!(B, A); copyto!(C, LinearIndices(A)); return B, C)
# end