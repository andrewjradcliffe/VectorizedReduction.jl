#
# Date created: 2022-03-25
# Author: aradclif
#
#
############################################################################################
# It is slightly more efficient to have a separate argmin, argmax,
# rather than lazily do vargmin(...) = vfindmin(...)[2].
# Alas! as vargminmax does not have a state for the minmax carried through
# the reduction block, the unroll/vectorization order can be different,
# and sadly, goes wrong. Until fixed, we shall compromise for ≈ 5% loss of performance
# and do it the lazy way.

vargminmax(f::F, op::OP, init::I, A, dims) where {F, OP, I} = vfindminmax(f, op, init, A, dims)[2]

# Convenience defintions
vargmax(f::F, A, dims) where {F<:Function} = vfindminmax(f, >, typemin, A, dims)[2]
vargmin(f::F, A, dims) where {F<:Function} = vfindminmax(f, <, typemax, A, dims)[2]
# ::AbstractArray required in order for kwargs interface to work
vargmax(A::AbstractArray, dims) = vargminmax(identity, >, typemin, A, dims)
vargmin(A::AbstractArray, dims) = vargminmax(identity, <, typemax, A, dims)

# Over all dims
vargmax(f::F, A) where {F<:Function} = vfindminmax(f, >, typemin, A, :)[2]
vargmin(f::F, A) where {F<:Function} = vfindminmax(f, <, typemax, A, :)[2]
# ::AbstractArray required in order for kwargs interface to work
vargmax(A::AbstractArray) = vargmax(identity, A)
vargmin(A::AbstractArray) = vargmin(identity, A)

# vargmax(f::F, A::AbstractVector) where {F<:Function} = vfindminmax(f, >, typemin, A, :)[2]
# vargmin(f::F, A::AbstractVector) where {F<:Function} = vfindminmax(f, <, typemax, A, :)[2]

# Provide inherently inefficient kwargs interface. Requires ::AbstractArray in the locations
# indicated above.
vargmax(f, A; dims=:) = vargmax(f, A, dims)
vargmax(A; dims=:) = vargmax(identity, A, dims)
vargmin(f, A; dims=:) = vargmin(f, A, dims)
vargmin(A; dims=:) = vargmin(identity, A, dims)

################
vtargminmax(f::F, op::OP, init::I, A, dims) where {F, OP, I} = vtfindminmax(f, op, init, A, dims)[2]

# Convenience defintions
vtargmax(f::F, A, dims) where {F<:Function} = vtfindminmax(f, >, typemin, A, dims)[2]
vtargmin(f::F, A, dims) where {F<:Function} = vtfindminmax(f, <, typemax, A, dims)[2]
# ::AbstractArray required in order for kwargs interface to work
vtargmax(A::AbstractArray, dims) = vtargminmax(identity, >, typemin, A, dims)
vtargmin(A::AbstractArray, dims) = vtargminmax(identity, <, typemax, A, dims)

# Over all dims
vtargmax(f::F, A) where {F<:Function} = vtfindminmax(f, >, typemin, A, :)[2]
vtargmin(f::F, A) where {F<:Function} = vtfindminmax(f, <, typemax, A, :)[2]
# ::AbstractArray required in order for kwargs interface to work
vtargmax(A::AbstractArray) = vtargmax(identity, A)
vtargmin(A::AbstractArray) = vtargmin(identity, A)

# Provide inherently inefficient kwargs interface. Requires ::AbstractArray in the locations
# indicated above.
vtargmax(f, A; dims=:) = vtargmax(f, A, dims)
vtargmax(A; dims=:) = vtargmax(identity, A, dims)
vtargmin(f, A; dims=:) = vtargmin(f, A, dims)
vtargmin(A; dims=:) = vtargmin(identity, A, dims)


################
# Varargs versions
vargmax(f::F, As::Vararg{AbstractArray, P}) where {F<:Function, P} =
    vfindminmax(f, >, typemin, As, :)[2]
vargmin(f::F, As::Vararg{AbstractArray, P}) where {F<:Function, P} =
    vfindminmax(f, <, typemax, As, :)[2]

vargmax(f, As::Vararg{AbstractArray, P}; dims=:) where {P} =
    vfindminmax(f, >, typemin, As, dims)[2]
vargmin(f, As::Vararg{AbstractArray, P}; dims=:) where {P} =
    vfindminmax(f, <, typemax, As, dims)[2]

vtargmax(f::F, As::Vararg{AbstractArray, P}) where {F<:Function, P} =
    vtargminmax(f, >, typemin, As, :)[2]
vtargmin(f::F, As::Vararg{AbstractArray, P}) where {F<:Function, P} =
    vtargminmax(f, <, typemax, As, :)[2]

vtargmax(f, As::Vararg{AbstractArray, P}; dims=:) where {P} =
    vtargminmax(f, >, typemin, As, dims)[2]
vtargmin(f, As::Vararg{AbstractArray, P}; dims=:) where {P} =
    vtargminmax(f, <, typemax, As, dims)[2]

################
# function vargminmax(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
#     Dᴬ = size(A)
#     Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
#     C = similar(A, Int, Dᴮ′)
#     _vargminmax!(f, op, init, C, A, dims)
#     return CartesianIndices(A)[C]
# end

# # Convenience defintions
# vargmax(f::F, A, dims) where {F<:Function} = vargminmax(f, >, typemin, A, dims)
# vargmin(f::F, A, dims) where {F<:Function} = vargminmax(f, <, typemax, A, dims)
# vargmax(A, dims) = vargminmax(identity, >, typemin, A, dims)
# vargmin(A, dims) = vargminmax(identity, <, typemax, A, dims)

# # over all dims
# @generated function vargminmax(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I, T, N}
#     opsym = OP.instance
#     initsym = I.instance
#     quote
#         m = $initsym(Base.promote_op(f, $T))
#         j = 0
#         @turbo for i ∈ eachindex(A)
#             newm = $opsym(f(A[i]), m)
#             m = ifelse(newm, f(A[i]), m)
#             j = ifelse(newm, i, j)
#         end
#         return CartesianIndices(A)[j]
#     end
# end

# vargmax(f::F, A) where {F<:Function} = vargminmax(f, >, typemin, A, :)
# vargmin(f::F, A) where {F<:Function} = vargminmax(f, <, typemax, A, :)
# vargmax(A) = vargmax(identity, A)#vargminmax(identity, >, typemin, A, :)
# vargmin(A) = vargmin(identity, A)#vargminmax(identity, <, typemax, A, :)

# function staticdim_argminmax_quote(OP, I, static_dims::Vector{Int}, N::Int)
#     A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
#     Cᵥ = Expr(:call, :view, :C)
#     Cᵥ′ = Expr(:ref, :Cᵥ)
#     rinds = Int[]
#     nrinds = Int[]
#     for d = 1:N
#         if d ∈ static_dims
#             push!(Cᵥ.args, Expr(:call, :firstindex, :C, d))
#             push!(rinds, d)
#         else
#             push!(Cᵥ.args, :)
#             push!(nrinds, d)
#             push!(Cᵥ′.args, Symbol(:i_, d))
#         end
#     end
#     reverse!(rinds)
#     reverse!(nrinds)
#     if !isempty(nrinds)
#         block = Expr(:block)
#         loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, C), $(nrinds[1]))), block)
#         for d ∈ @view(nrinds[2:end])
#             newblock = Expr(:block)
#             push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, C), $d)), newblock))
#             block = newblock
#         end
#         rblock = block
#         # Pre-reduction
#         ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :T))
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
#         cmpr = Expr(:(=), :newminmax, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
#         push!(block.args, cmpr)
#         setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newminmax, Expr(:call, :f, A), :ξ))
#         push!(block.args, setmax)
#         for d ∈ rinds
#             setj = Expr(:(=), Symbol(:j_, d),
#                         Expr(:call, :ifelse, :newminmax, Symbol(:i_, d), Symbol(:j_, d)))
#             push!(block.args, setj)
#         end
#         # Push to after reduction loop
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
#         push!(rblock.args, Expr(:(=), Cᵥ′, setc))
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
#             Cᵥ = $Cᵥ
#             @turbo $loops
#             return C
#         end
#     else
#         # Pre-reduction
#         ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :T))
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
#         cmpr = Expr(:(=), :newminmax, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
#         push!(block.args, cmpr)
#         setmax = Expr(:(=), :ξ, Expr(:call, :ifelse, :newminmax, Expr(:call, :f, A), :ξ))
#         push!(block.args, setmax)
#         for d ∈ rinds
#             setj = Expr(:(=), Symbol(:j_, d),
#                         Expr(:call, :ifelse, :newminmax, Symbol(:i_, d), Symbol(:j_, d)))
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
#             Cᵥ = $Cᵥ
#             $ξ
#             @turbo $loops
#             Cᵥ[] = $setc
#             return C
#         end
#     end
# end

# function branches_argminmax_quote(OP, I, N::Int, M::Int, D)
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
#                 qnew = Expr(ifsym, :(dimm == $n), :(return _vargminmax!(f, op, init, C, A, $tc)))
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
#             push!(qold.args, Expr(:block, :(return _vargminmax!(f, op, init, C, A, $tc))))
#             return q
#         end
#     end
#     return staticdim_argminmax_quote(OP, I, static_dims, N)
# end

# @generated function _vargminmax!(f::F, op::OP, init::I, C::AbstractArray{Tₗ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₗ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
#     branches_argminmax_quote(OP, I, N, M, D)
# end
# @generated function _vargminmax!(f::F, op::OP, init::I, C::AbstractArray{Tₗ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₗ, T, N}
#     :(copyto!(C, LinearIndices(A)); return C)
# end
