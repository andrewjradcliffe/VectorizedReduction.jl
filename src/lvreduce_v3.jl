#
# Date created: 2022-03-22
# Author: aradclif
#
#
############################################################################################
# - Implementing reduce(op, ...) as simply mapreduce(identity, op, ...)
# - Using compile-time branch resolution rather than dispatch system (well,
#   less on the dispatch system)

_dim(::Type{StaticInt{N}}) where {N} = N::Int

# Demonstrated: that technically, `f`, can be anonymous. The reduction still needs
# to be a regular binary operation which is known to LoopVectorization.
# Hence, it only seems worthwhile to not force specialization on f.
# Moreover, it is always a slight advantage to use a named function over an anonymous...
# In fact, having changed it over, it is a beautiful thing -- same behavior as Julia Base.

function vvmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    # B = similar(A, Base.promote_op(op, T, Int), Dᴮ′)
    B = similar(A, Base.promote_op(op, Base.promote_op(f, T), Int), Dᴮ′)
    _vvmapreduce!(f, op, init, B, A, dims)
    # One must ensure that the dims are sorted, as the type dispatch within and on
    # the generated branch functions AND work functions respects the order of permutations.
    # Consequently, given some N and M, in the most extreme case, a naive approach would
    # generate new functions for all the possible permutations which
    # could be generated from the combinations N choose (M - 1),
    # i.e. N! / (N - M + 1)! ≡ factorial(N) / factorial(N - M + 1)
    # M - 1 arises due to the fact that the Mᵗʰ branch of the function will
    # result in a work function call.
    # Conversely, if one ensures that permutations do not matter (by sorting dims),
    # then at most N choose M work functions will be compiled.
    # Alas, one is loathe to ensure that the dimensions will be sorted, as
    # it incurs overhead of approximately 90ns, which, for length(A) = 625,
    # is substantial (increase in time by factor of 1.33, and 80 bytes).
    # Perhaps one could just make it the caller's responsibility to avoid
    # this unnecessary compilation.
    # v = sort!(collect(dims))
    # newdims = ntuple(i -> v[i], Val(M))
    # _vvmapreduce!(f, op, init, B, A, newdims)
    return B
end

# dims determination would ideally be non-allocating. Also, who would
# call this anyway? Almost assuredly, a caller would already know dims, hence
# just call _vvmapreduce! anyway.
function vvmapreduce!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}) where {F, OP, I, Tₒ, T, N}
    Dᴬ = size(A)
    Dᴮ = size(B)
    dims = Tuple((d for d ∈ eachindex(Dᴮ) if isone(Dᴮ[d])))
    all(d -> Dᴮ[d] == Dᴬ[d], (d for d ∈ eachindex(Dᴮ) if !isone(Dᴮ[d]))) || throw(DimensionMismatch)
    _vvmapreduce!(f, op, init, B, A, dims)
    return B
end

# Convenience definitions
vvsum(f::F, A, dims) where {F} = vvmapreduce(f, +, zero, A, dims)
vvprod(f::F, A, dims) where {F} = vvmapreduce(f, *, one, A, dims)
vvmaximum(f::F, A, dims) where {F} = vvmapreduce(f, max, typemin, A, dims)
vvminimum(f::F, A, dims) where {F} = vvmapreduce(f, min, typemax, A, dims)

vvsum(A, dims) = vvmapreduce(identity, +, zero, A, dims)
vvprod(A, dims) = vvmapreduce(identity, *, one, A, dims)
vvmaximum(A, dims) = vvmapreduce(identity, max, typemin, A, dims)
vvminimum(A, dims) = vvmapreduce(identity, min, typemax, A, dims)

# reduction over all dims
@generated function vvmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I, T, N}
    # fsym = F.instance
    opsym = OP.instance
    initsym = I.instance
    # Tₒ = Base.promote_op(opsym, Base.promote_op(fsym, T), Int)
    quote
        # ξ = $initsym($Tₒ)
        ξ = $initsym(Base.promote_op($opsym, Base.promote_op(f, $T), Int))
        @turbo for i ∈ eachindex(A)
            ξ = $opsym(f(A[i]), ξ)
        end
        return ξ
    end
end

# The dispatch on function type is faster if the function is named,
# slower (≈ 15%) for anonymous. This would only affect performance in the REPL,
# hence, it's not really an issue.
vvsum(f::F, A) where {F<:Function} = vvmapreduce(f, +, zero, A, :)
vvprod(f::F, A) where {F<:Function} = vvmapreduce(f, *, one, A, :)
vvmaximum(f::F, A) where {F<:Function} = vvmapreduce(f, max, typemin, A, :)
vvminimum(f::F, A) where {F<:Function} = vvmapreduce(f, min, typemax, A, :)

vvsum(A) = vvmapreduce(identity, +, zero, A, :)
vvprod(A) = vvmapreduce(identity, *, one, A, :)
vvmaximum(A) = vvmapreduce(identity, max, typemin, A, :)
vvminimum(A) = vvmapreduce(identity, min, typemax, A, :)

# A surprising convenience opportunity -- albeit, a specific implementation will
# be necessary in order to utilize Array{Bool} rather than the default, which
# will get promoted to Array{Int}. Oddly, only works on Array{<:Integer} inputs.
# A workaround (inefficient) would be to sum Bools, then compare to length
# vany(f::F, A, dims) where {F} = vvmapreduce(f, |, zero, A, dims)
# vall(f::F, A, dims) where {F} = vvmapreduce(f, &, one, A, dims)
# vany(A, dims) = vvmapreduce(identity, |, zero, A, dims)
# vall(A, dims) = vvmapreduce(identity, &, one, A, dims)

# A curious solution: initialize with a value, rather than `zero` or `one`.
# Most likely, this is due promotion by LoopVectorization, which in this case
# breaks the ability to use anything but Array{<:Integer} for a Boolean predicate
# function. However, initializing to a Bool solves this, preventing any
# problems with the type conversion.
# Alas! Another issue which also happens to come up with findminmax, argminmax:
# Reductions on the first dimension are not well-defined, hence, the same
# errors occur. Awkwardly, the same fix could be used: re-shape to shift
# everything to +1 dimension, reduce, then drop the dimension.
# Unfortunately, this results in worse performance than just calling
# `any` from Julia Base. It would probably be faster to do vcount
# and then compare...
function vany(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Bool, Dᴮ′)
    _vvmapreduce_init!(f, |, false, B, A, dims)
    return B
end
function vall(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Bool, Dᴮ′)
    _vvmapreduce_init!(f, &, true, B, A, dims)
    return B
end
function vcount(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Int, Dᴮ′)
    _vvmapreduce!(f, +, zero, B, A, dims)
    return B
end

vany(A, dims) = vany(identity, A, dims)
vall(A, dims) = vall(identity, A, dims)
# # An approach to permitting anonymous functions
# function _named(f::F) where {F<:Function}
#     F <: Base.Fix2 ? begin @eval function $(gensym())(y) $(f.f)(y, $(f.x)) end end : f
# end

# function vany2(f::F, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, T, N, M}
#     if 1 ∈ dims
#         newdims = ntuple(d -> dims[d] + 1, Val(M))
#         Dᴬ′ = ntuple(d -> d == 1 ? 1 : size(A, d - 1), Val(N+1))
#         B = vany(f, reshape(A, Dᴬ′), newdims)
#         return dropdims(B, dims=1)
#     else
#         return vany(f, A, dims)
#     end
# end

# A silly definition, but it is nonetheless possible... and, an improvement by 2x
function vmask(f::F, A::AbstractArray{T, N}) where {F, T, N}
    B = similar(A, Bool)
    @turbo for i ∈ eachindex(A)
        B[i] = f(A[i])
    end
    return B
end
vfindall(f::F, A::AbstractArray{T, N}) where {F, T, N} = CartesianIndices(A)[vmask(f, A)]

# Define reduce
vvreduce(op::OP, init::I, A, dims) where {OP, I} = vvmapreduce(identity, op, init, A, dims)

function staticdim_mapreduce_quote(OP, I, static_dims::Vector{Int}, N::Int)
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
        loops = Expr(:for, Expr(:(=), Symbol(:i_, nrinds[1]),
                                Expr(:call, :indices, Expr(:tuple, :A, :B), nrinds[1])), block)
        # loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B), $(nrinds[1]))), block)
        for i = 2:length(nrinds)
            # for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args,
                  Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
                                  Expr(:call, :indices, Expr(:tuple, :A, :B), nrinds[i])), newblock))
            # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :axes, :A, d)), newblock))
            # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance),
                                   Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            Bᵥ = $Bᵥ
            @turbo $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, Expr(:(=), Symbol(:i_, rinds[1]),
                                Expr(:call, :axes, :A, rinds[1])), block)
        # loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for i = 2:length(rinds)
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
                                              Expr(:call, :axes, :A, rinds[i])), newblock))
            # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance),
                                   Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        return quote
            Bᵥ = $Bᵥ
            $ξ
            @turbo $loops
            Bᵥ[] = ξ
            return B
        end
    end
end

function branches_mapreduce_quote(OP, I, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vvmapreduce!(f, op, init, B, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vvmapreduce!(f, op, init, B, A, $tc))))
            return q
        end
    end
    return staticdim_mapreduce_quote(OP, I, static_dims, N)
end

@generated function _vvmapreduce!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_mapreduce_quote(OP, I, N, M, D)
end
@generated function _vvmapreduce!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N}
    :(copyto!(B, A); return B)
end

################
# Version wherein an initial value is supplied

function vvmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I<:Number, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(op, Base.promote_op(f, T), Int), Dᴮ′)
    _vvmapreduce_init!(f, op, init, B, A, dims)
    return B
end

# reduction over all dims
@generated function vvmapreduce(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I<:Number, T, N}
    # fsym = F.instance
    opsym = OP.instance
    # Tₒ = Base.promote_op(opsym, Base.promote_op(fsym, T), Int)
    quote
        # ξ = convert($Tₒ, init)
        ξ = convert(Base.promote_op($opsym, Base.promote_op(f, $T), Int), init)
        @turbo for i ∈ eachindex(A)
            # ξ = $opsym($fsym(A[i]), ξ)
            ξ = $opsym(f(A[i]), ξ)
        end
        return ξ
    end
end

function staticdim_mapreduce_init_quote(OP, static_dims::Vector{Int}, N::Int)
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
        # ξ₀ = Expr(:call, Expr(:call, :eltype, :Bᵥ), :init)
        ξ₀ = Expr(:call, :convert, Expr(:call, :eltype, :Bᵥ), :init)
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
        # Pre-reduction
        ξ = Expr(:(=), :ξ, :ξ₀)
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :axes, :A, d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance),
                                   Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        return quote
            Bᵥ = $Bᵥ
            ξ₀ = $ξ₀
            @turbo $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, Expr(:call, :eltype, :Bᵥ), :init))
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
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance),
                                   Expr(:call, :f, A), :ξ))
        push!(block.args, setξ)
        return quote
            Bᵥ = $Bᵥ
            $ξ
            @turbo $loops
            Bᵥ[] = ξ
            return B
        end
    end
end

function branches_mapreduce_init_quote(OP, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vvmapreduce_init!(f, op, init, B, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vvmapreduce_init!(f, op, init, B, A, $tc))))
            return q
        end
    end
    return staticdim_mapreduce_init_quote(OP, static_dims, N)
end

@generated function _vvmapreduce_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_mapreduce_init_quote(OP, N, M, D)
end
@generated function _vvmapreduce_init!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N}
    :(copyto!(B, A); return B)
end


################
# function vvmapreduce2(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
#     Dᴬ = size(A)
#     Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
#     B = similar(A, Base.promote_op(op, Base.promote_op(f, T), Int), Dᴮ′)
#     _vvmapreduce2!(f, op, init, B, A, dims)
#     return B
# end
# # reduction over all dims
# @generated function vvmapreduce2(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I, T, N}
#     fsym = F.instance
#     opsym = OP.instance
#     initsym = I.instance
#     # Tₒ = Base.promote_op(opsym, Base.promote_op(fsym, T), Int)
#     quote
#         # ξ = $initsym($Tₒ)
#         ξ = $initsym(Base.promote_op($opsym, Base.promote_op(f, $T), Int))
#         @turbo for i ∈ eachindex(A)
#             ξ = $opsym(f(A[i]), ξ)
#         end
#         return ξ
#     end
# end

# function staticdim_mapreduce2_quote(OP, I, static_dims::Vector{Int}, N::Int)
#     A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
#     Bᵥ = Expr(:call, :view, :B)
#     Bᵥ′ = Expr(:ref, :Bᵥ)
#     rinds = Int[]
#     nrinds = Int[]
#     for d = 1:N
#         if d ∈ static_dims
#             push!(Bᵥ.args, Expr(:call, :firstindex, :B, d))
#             push!(rinds, d)
#         else
#             push!(Bᵥ.args, :)
#             push!(nrinds, d)
#             push!(Bᵥ′.args, Symbol(:i_, d))
#         end
#     end
#     reverse!(rinds)
#     reverse!(nrinds)
#     if !isempty(nrinds)
#         block = Expr(:block)
#         loops = Expr(:for, Expr(:(=), Symbol(:i_, nrinds[1]),
#                                 Expr(:call, :indices, Expr(:tuple, :A, :B), nrinds[1])), block)
#         # loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B), $(nrinds[1]))), block)
#         for i = 2:length(nrinds)
#             # for d ∈ @view(nrinds[2:end])
#             newblock = Expr(:block)
#             push!(block.args,
#                   Expr(:for, Expr(:(=), Symbol(:i_, nrinds[i]),
#                                   Expr(:call, :indices, Expr(:tuple, :A, :B), nrinds[i])), newblock))
#             # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B), $d)), newblock))
#             block = newblock
#         end
#         rblock = block
#         # Pre-reduction
#         ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
#         push!(rblock.args, ξ)
#         # Reduction loop
#         for d ∈ rinds
#             newblock = Expr(:block)
#             push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, d), Expr(:call, :axes, :A, d)), newblock))
#             # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
#             block = newblock
#         end
#         # Push to inside innermost loop
#         setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance),
#                                    Expr(:call, :f, A), :ξ))
#         push!(block.args, setξ)
#         setb = Expr(:(=), Bᵥ′, :ξ)
#         push!(rblock.args, setb)
#         return quote
#             Bᵥ = $Bᵥ
#             @turbo $loops
#             return B
#         end
#     else
#         # Pre-reduction
#         ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
#         # Reduction loop
#         block = Expr(:block)
#         loops = Expr(:for, Expr(:(=), Symbol(:i_, rinds[1]),
#                                 Expr(:call, :axes, :A, rinds[1])), block)
#         # loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
#         for i = 2:length(rinds)
#             newblock = Expr(:block)
#             push!(block.args, Expr(:for, Expr(:(=), Symbol(:i_, rinds[i]),
#                                               Expr(:call, :axes, :A, rinds[i])), newblock))
#             # push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
#             block = newblock
#         end
#         # Push to inside innermost loop
#         setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance),
#                                    Expr(:call, :f, A), :ξ))
#         push!(block.args, setξ)
#         return quote
#             Bᵥ = $Bᵥ
#             $ξ
#             @turbo $loops
#             Bᵥ[] = ξ
#             return B
#         end
#     end
# end

# function branches_mapreduce2_quote(OP, I, N::Int, M::Int, D)
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
#                 qnew = Expr(ifsym, :(dimm == $n), :(return _vvmapreduce2!(f, op, init, B, A, $tc)))
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
#             push!(qold.args, Expr(:block, :(return _vvmapreduce2!(f, op, init, B, A, $tc))))
#             return q
#         end
#     end
#     return staticdim_mapreduce2_quote(OP, I, static_dims, N)
# end

# @generated function _vvmapreduce2!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
#     branches_mapreduce2_quote(OP, I, N, M, D)
# end
# @generated function _vvmapreduce2!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, T, N}
#     :(copyto!(B, A); return B)
# end
