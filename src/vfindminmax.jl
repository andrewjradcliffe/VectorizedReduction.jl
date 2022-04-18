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
vfindminmax(f, op, init, A, dims::Int) = vfindminmax(f, op, init, A, (dims,))

"""
    vfindmax(f, A::AbstractArray, dims=:) -> (f(x), index)

Return the value and the index of the argument which maximizes `f` over the
dimensions `dims`, which may be `::Int`, `::NTuple{M, Int} where {M}` or `::Colon`.
Expands upon the functionality provided in Julia Base.

# Additional Notes
Due to the current limitations of LoopVectorization, searches over the first dimension
of an array are not well-supported. A workaround is possible by reshaping `A` but
the resultant performance is often only on par with `findmax`. As a temporary convenience,
`findmax1` is provided for explicit uses of the re-shaping strategy, though the user
is cautioned as to the performance problems.
"""
vfindmax(f::F, A, dims) where {F<:Function} = vfindminmax(f, >, typemin, A, dims)

"""
    vfindmin(f, A::AbstractArray, dims=:) -> (f(x), index)

Return the value and the index of the argument which minimizes `f` over the
dimensions `dims`, which may be `::Int`, `::NTuple{M, Int} where {M}` or `::Colon`.
Expands upon the functionality provided in Julia Base.

# Additional Notes
Due to the current limitations of LoopVectorization, searches over the first dimension
of an array are not well-supported. A workaround is possible by reshaping `A` but
the resultant performance is often only on par with `findmin`. As a temporary convenience,
`findmin1` is provided for explicit uses of the re-shaping strategy, though the user
is cautioned as to the performance problems.
"""
vfindmin(f::F, A, dims) where {F<:Function} = vfindminmax(f, <, typemax, A, dims)

# ::AbstractArray required in order for kwargs interface to work

"""
    vfindmax(A::AbstractArray, dims=:) -> (x, index)

Return the maximal element and its index over the dimensions `dims`.
"""
vfindmax(A::AbstractArray, dims) = vfindminmax(identity, >, typemin, A, dims)

"""
    vfindmin(A::AbstractArray, dims=:) -> (x, index)

Return the minimal element and its index over the dimensions `dims`.
"""
vfindmin(A::AbstractArray, dims) = vfindminmax(identity, <, typemax, A, dims)

# over all dims
@generated function vfindminmax(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I, T, N}
    opsym = OP.instance
    initsym = I.instance
    quote
        m = $initsym(Base.promote_op(f, $T))
        j = 1
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
        j = 1
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
# ::AbstractArray required in order for kwargs interface to work
vfindmax(A::AbstractArray) = vfindmax(identity, A)
vfindmin(A::AbstractArray) = vfindmin(identity, A)

"""
    vfindmax(f, A; dims) -> (f(x), index)
    vfindmax(A; dims) -> (x, index)

Identical to non-keywords args version; slightly less performant due to use of kwargs.
"""
vfindmax(f, A; dims=:) = vfindmax(f, A, dims)
vfindmax(A; dims=:) = vfindmax(identity, A, dims)

"""
    vfindmin(f, A; dims) -> (f(x), index)
    vfindmin(A; dims) -> (x, index)

Identical to non-keywords args versions; slightly less performant due to use of kwargs.
"""
vfindmin(f, A; dims=:) = vfindmin(f, A, dims)
vfindmin(A; dims=:) = vfindmin(identity, A, dims)

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
        cmpr = Expr(:(=), :newm, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, cmpr)
        setm = Expr(:(=), :ξ, Expr(:call, :ifelse, :newm, Expr(:call, :f, A), :ξ))
        push!(block.args, setm)
        for d ∈ rinds
            setj = Expr(:(=), Symbol(:j_, d),
                        Expr(:call, :ifelse, :newm, Symbol(:i_, d), Symbol(:j_, d)))
            # setj = :($(Symbol(:j_, d)) = ifelse(newm, $(Symbol(:i_, d)), $(Symbol(:j_, d))))
            push!(block.args, setj)
        end
        # Push to after reduction loop
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        # Potential loop-carried dependency
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)Iₖ    : I₁ + D₁I₂ + D₁D₂I₃ + ⋯ + D₁D₂⋯Dₖ₋₁Iₖ
        # # The less efficient version (good for visualizing, though)
        # setc = Expr(:call, :+)
        # for d ∈ rinds
        #     push!(setc.args, d == 1 ? :j_1 :
        #         Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
        # end
        # for d ∈ nrinds
        #     push!(setc.args, d == 1 ? :i_1 :
        #         Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:i_, d)))
        # end
        # # These complete the expression: 1 + ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)(Iₖ - 1)
        # push!(setc.args, 1, :Dstar)
        # postj = Expr(:(=), Cᵥ′, setc)
        # push!(rblock.args, postj)
        # # strides, offsets
        t = Expr(:tuple)
        for d = 1:N
            push!(t.args, Symbol(:D_, d))
        end
        sz = Expr(:(=), t, Expr(:call, :size, :A))
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)    : 1 + D₁ + D₁D₂ + ⋯ + D₁D₂⋯Dₖ₋₁
        dstar = Expr(:call, :+, 1)
        for d = 2:N
            push!(dstar.args, d == 2 ? :D_1 : Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
        end
        # Version which precomputes the unchanging components of setc.
        tl = Expr(:tuple)
        tr = Expr(:tuple)
        for d = 3:N
            push!(tl.args, Symbol(:D_, ntuple(identity, d - 1)...))
            push!(tr.args, Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
        end
        setc = Expr(:call, :+)
        for d ∈ rinds
            push!(setc.args, d == 1 ? :j_1 :
                Expr(:call, :*, Symbol(:D_, ntuple(identity, d - 1)...), Symbol(:j_, d)))
        end
        for d ∈ nrinds
            push!(setc.args, d == 1 ? :i_1 :
                Expr(:call, :*, Symbol(:D_, ntuple(identity, d - 1)...), Symbol(:i_, d)))
        end
        push!(setc.args, 1, :Dstar)
        postj = Expr(:(=), Cᵥ′, setc)
        push!(rblock.args, postj)
        return quote
            $sz
            $tl = $tr
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
        cmpr = Expr(:(=), :newm, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, cmpr)
        setm = Expr(:(=), :ξ, Expr(:call, :ifelse, :newm, Expr(:call, :f, A), :ξ))
        push!(block.args, setm)
        for d ∈ rinds
            setj = Expr(:(=), Symbol(:j_, d),
                        Expr(:call, :ifelse, :newm, Symbol(:i_, d), Symbol(:j_, d)))
            # setj = :($(Symbol(:j_, d)) = ifelse(newm, $(Symbol(:i_, d)), $(Symbol(:j_, d))))
            push!(block.args, setj)
        end
        # The less efficient version (good for visualizing, though). It does not matter
        # here, as this just handles the reduction over all dimensions specified via dims.
        # In other words, the computation of the linear index only happens once.
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)Iₖ    : I₁ + D₁I₂ + D₁D₂I₃ + ⋯ + D₁D₂⋯Dₖ₋₁Iₖ
        setc = Expr(:call, :+)
        for d ∈ rinds
            push!(setc.args, d == 1 ? :j_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
        end
        for d ∈ nrinds
            push!(setc.args, d == 1 ? :i_1 :
                Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:i_, d)))
        end
        # These complete the expression: 1 + ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)(Iₖ - 1)
        push!(setc.args, 1, :Dstar)
        # strides, offsets
        t = Expr(:tuple)
        for d = 1:N
            push!(t.args, Symbol(:D_, d))
        end
        sz = Expr(:(=), t, Expr(:call, :size, :A))
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)    : 1 + D₁ + D₁D₂ + ⋯ + D₁D₂⋯Dₖ₋₁
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

# In the case of rinds = ∅, this just corresponds to a map
@generated function _vfindminmax!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₗ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, Tₗ, T, N}
    :(vvmap!(f, B, A); copyto!(C, LinearIndices(A)); return B, C)
end

############################################################################################
function vtfindminmax(f::F, op::OP, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, I, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(f, T), Dᴮ′)
    C = similar(A, Int, Dᴮ′)
    _vtfindminmax!(f, op, init, B, C, A, dims)
    return B, CartesianIndices(A)[C]
end
vtfindminmax(f, op, init, A, dims::Int) = vtfindminmax(f, op, init, A, (dims,))


"""
    vtfindmax(f, A::AbstractArray, dims=:) -> (f(x), index)

Return the value and the index of the argument which maximizes `f` over the
dimensions `dims`, which may be `::Int`, `::NTuple{M, Int} where {M}` or `::Colon`.
Threaded.

See also: [`vfindmax`](@ref)
"""
vtfindmax(f::F, A, dims) where {F<:Function} = vtfindminmax(f, >, typemin, A, dims)

"""
    vtfindmin(f, A::AbstractArray, dims=:) -> (f(x), index)

Return the value and the index of the argument which minimizes `f` over the
dimensions `dims`, which may be `::Int`, `::NTuple{M, Int} where {M}` or `::Colon`.
Threaded.

See also: [`vfindmin`](@ref)
"""
vtfindmin(f::F, A, dims) where {F<:Function} = vtfindminmax(f, <, typemax, A, dims)
# ::AbstractArray required in order for kwargs interface to work
"""
    vtfindmax(A::AbstractArray, dims=:) -> (x, index)

Return the maximal element and its index over the dimensions `dims`. Threaded.
"""
vtfindmax(A::AbstractArray, dims) = vtfindminmax(identity, >, typemin, A, dims)

"""
    vtfindmin(A::AbstractArray, dims=:) -> (x, index)

Return the minimal element and its index over the dimensions `dims`. Threaded.
"""
vtfindmin(A::AbstractArray, dims) = vtfindminmax(identity, <, typemax, A, dims)

# over all dims
@generated function vtfindminmax(f::F, op::OP, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, I, T, N}
    opsym = OP.instance
    initsym = I.instance
    quote
        m = $initsym(Base.promote_op(f, $T))
        j = 1
        @tturbo for i ∈ eachindex(A)
            newm = $opsym(f(A[i]), m)
            m = ifelse(newm, f(A[i]), m)
            j = ifelse(newm, i, j)
        end
        return m, CartesianIndices(A)[j]
    end
end

# A seemingly superfluous dispatch, but is necessary to enforce that a vector returns
# a linear index.
@generated function vtfindminmax(f::F, op::OP, init::I, A::AbstractVector{T}, ::Colon) where {F, OP, I, T}
    opsym = OP.instance
    initsym = I.instance
    quote
        m = $initsym(Base.promote_op(f, $T))
        j = 1
        @tturbo for i ∈ eachindex(A)
            newm = $opsym(f(A[i]), m)
            m = ifelse(newm, f(A[i]), m)
            j = ifelse(newm, i, j)
        end
        return m, j
    end
end

vtfindmax(f::F, A) where {F<:Function} = vtfindminmax(f, >, typemin, A, :)
vtfindmin(f::F, A) where {F<:Function} = vtfindminmax(f, <, typemax, A, :)
# ::AbstractArray required in order for kwargs interface to work
vtfindmax(A::AbstractArray) = vtfindmax(identity, A)
vtfindmin(A::AbstractArray) = vtfindmin(identity, A)

# Provide inherently inefficient kwargs interface. Requires ::AbstractArray in the locations
# indicated above.

"""
    vtfindmax(f, A; dims) -> (f(x), index)
    vtfindmax(A; dims) -> (x, index)

Identical to non-keywords args version; slightly less performant due to use of kwargs. Threaded.
"""
vtfindmax(f, A; dims=:) = vtfindmax(f, A, dims)
vtfindmax(A; dims=:) = vtfindmax(identity, A, dims)

"""
    vtfindmin(f, A; dims) -> (f(x), index)
    vtfindmin(A; dims) -> (x, index)

Identical to non-keywords args version; slightly less performant due to use of kwargs. Threaded.
"""
vtfindmin(f, A; dims=:) = vtfindmin(f, A, dims)
vtfindmin(A; dims=:) = vtfindmin(identity, A, dims)

function staticdim_tfindminmax_quote(OP, I, static_dims::Vector{Int}, N::Int)
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
        # cmpr = Expr(:(=), :newm, Expr(:call, :(>), A, :ξ))
        cmpr = Expr(:(=), :newm, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, cmpr)
        # setm = Expr(:(=), :ξ, Expr(:call, :ifelse, :newm, A, :ξ))
        setm = Expr(:(=), :ξ, Expr(:call, :ifelse, :newm, Expr(:call, :f, A), :ξ))
        push!(block.args, setm)
        for d ∈ rinds
            setj = Expr(:(=), Symbol(:j_, d),
                        Expr(:call, :ifelse, :newm, Symbol(:i_, d), Symbol(:j_, d)))
            # setj = :($(Symbol(:j_, d)) = ifelse(newm, $(Symbol(:i_, d)), $(Symbol(:j_, d))))
            push!(block.args, setj)
        end
        # Push to after reduction loop
        setb = Expr(:(=), Bᵥ′, :ξ)
        push!(rblock.args, setb)
        # Potential loop-carried dependency
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)Iₖ    : I₁ + D₁I₂ + D₁D₂I₃ + ⋯ + D₁D₂⋯Dₖ₋₁Iₖ
        # # The less efficient version (good for visualizing, though)
        # setc = Expr(:call, :+)
        # for d ∈ rinds
        #     push!(setc.args, d == 1 ? :j_1 :
        #         Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:j_, d)))
        # end
        # for d ∈ nrinds
        #     push!(setc.args, d == 1 ? :i_1 :
        #         Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)..., Symbol(:i_, d)))
        # end
        # push!(setc.args, 1, :Dstar)
        # postj = Expr(:(=), Cᵥ′, setc)
        # push!(rblock.args, postj)
        # # strides, offsets
        t = Expr(:tuple)
        for d = 1:N
            push!(t.args, Symbol(:D_, d))
        end
        sz = Expr(:(=), t, Expr(:call, :size, :A))
        dstar = Expr(:call, :+, 1)
        for d = 2:N
            push!(dstar.args, d == 2 ? :D_1 : Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
        end
        # Version which precomputes the unchanging components of setc.
        tl = Expr(:tuple)
        tr = Expr(:tuple)
        for d = 3:N
            push!(tl.args, Symbol(:D_, ntuple(identity, d - 1)...))
            push!(tr.args, Expr(:call, :*, ntuple(i -> Symbol(:D_, i), d - 1)...))
        end
        setc = Expr(:call, :+)
        for d ∈ rinds
            push!(setc.args, d == 1 ? :j_1 :
                Expr(:call, :*, Symbol(:D_, ntuple(identity, d - 1)...), Symbol(:j_, d)))
        end
        for d ∈ nrinds
            push!(setc.args, d == 1 ? :i_1 :
                Expr(:call, :*, Symbol(:D_, ntuple(identity, d - 1)...), Symbol(:i_, d)))
        end
        push!(setc.args, 1, :Dstar)
        postj = Expr(:(=), Cᵥ′, setc)
        push!(rblock.args, postj)
        return quote
            $sz
            $tl = $tr
            Dstar = $dstar
            Dstar = -Dstar
            Bᵥ = $Bᵥ
            Cᵥ = $Cᵥ
            @tturbo $loops
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
        cmpr = Expr(:(=), :newm, Expr(:call, Symbol(OP.instance), Expr(:call, :f, A), :ξ))
        push!(block.args, cmpr)
        setm = Expr(:(=), :ξ, Expr(:call, :ifelse, :newm, Expr(:call, :f, A), :ξ))
        push!(block.args, setm)
        for d ∈ rinds
            setj = Expr(:(=), Symbol(:j_, d),
                        Expr(:call, :ifelse, :newm, Symbol(:i_, d), Symbol(:j_, d)))
            # setj = :($(Symbol(:j_, d)) = ifelse(newm, $(Symbol(:i_, d)), $(Symbol(:j_, d))))
            push!(block.args, setj)
        end
        # The less efficient version (good for visualizing, though). It does not matter
        # here, as this just handles the reduction over all dimensions specified via dims.
        # In other words, the computation of the linear index only happens once.
        # ∑ₖ₌₁ᴺ(∏ᵢ₌₁ᵏ⁻¹Dᵢ)Iₖ    : I₁ + D₁I₂ + D₁D₂I₃ + ⋯ + D₁D₂⋯Dₖ₋₁Iₖ
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
            @tturbo $loops
            Bᵥ[] = ξ
            Cᵥ[] = $setc
            return B, C
        end
    end
end

function branches_tfindminmax_quote(OP, I, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtfindminmax!(f, op, init, B, C, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vtfindminmax!(f, op, init, B, C, A, $tc))))
            return q
        end
    end
    return staticdim_tfindminmax_quote(OP, I, static_dims, N)
end

@generated function _vtfindminmax!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₗ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, I, Tₒ, Tₗ, T, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_tfindminmax_quote(OP, I, N, M, D)
end
@generated function _vtfindminmax!(f::F, op::OP, init::I, B::AbstractArray{Tₒ, N}, C::AbstractArray{Tₗ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, I, Tₒ, Tₗ, T, N}
    :(vtmap!(f, B, A); copyto!(C, LinearIndices(A)); return B, C)
end
