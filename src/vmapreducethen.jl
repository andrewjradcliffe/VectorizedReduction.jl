#
# Date created: 2022-09-13
# Author: aradclif
#
#
############################################################################################

"""
    vmapreducethen(f, op, g, A::AbstractArray; dims=:, init)

Apply function `f` to each element of `A`, reduce the result over the dimensions
`dims` using the binary function `op`, then apply `g` to the result. Equivalent to
`g.(mapreduce(f, op, A, dims=dims, init=init))` but avoids the intermediate implied
by said expression while also fusing the post-transform `g` such that the output array
is populated in a single pass.

The reduction necessitates an initial value `init` which may be `<:Number` or a function
which accepts a single type argument (e.g. `zero`);
`init` is optional for binary operators `+`, `*`, `min`, and `max`.

# Examples
```jldoctest
julia> vmapreducethen(abs2, +, √, [1 2; 3 4], dims=1)    # L₂-norm; see `vnorm`
1×2 Matrix{Float64}:
 3.16228  4.47214

julia> vmapreducethen(abs2, +, √, [1 2; 3 4], dims=2, init=1000.0)
2×1 Matrix{Float64}:
 31.701734968294716
 32.01562118716424

julia> vmapreducethen(exp, +, log, [5 6; 7 8], dims=1)    # LSE, but recommend `vlogsumexp`
1×2 Matrix{Float64}:
 7.12693  8.12693
```
"""
function vmapreducethen(f::F, op::OP, g::G, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, G, I, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(g, Base.promote_op(op, Base.promote_op(f, T), Int)), Dᴮ′)
    _vmapreducethen!(f, op, g, init, B, A, dims)
    return B
end
vmapreducethen(f, op, g, init, A, dims::Int) = vmapreducethen(f, op, g, init, A, (dims,))

function vmapreducethen!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}) where {F, OP, G, I, Tₒ, T, N}
    Dᴬ = size(A)
    Dᴮ = size(B)
    dims = Tuple((d for d ∈ eachindex(Dᴮ) if isone(Dᴮ[d])))
    all(d -> Dᴮ[d] == Dᴬ[d], (d for d ∈ eachindex(Dᴮ) if !isone(Dᴮ[d]))) || throw(DimensionMismatch)
    _vmapreducethen!(f, op, g, init, B, A, dims)
    return B
end

@generated function vmapreducethen(f::F, op::OP, g::G, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, G, I, T, N}
    opsym = OP.instance
    initsym = I.instance
    quote
        ξ = $initsym(Base.promote_op($opsym, Base.promote_op(f, $T), Int))
        @turbo check_empty=true for i ∈ eachindex(A)
            ξ = $opsym(ξ, f(A[i]))
        end
        return g(ξ)
    end
end
vmapreducethen(f::F, op::OP, g::G, init::I, A::AbstractArray) where {F, OP, G, I} = vmapreducethen(f, op, g, init, A, :)

for (op, init) ∈ zip((:+, :*, :max,:min), (:zero, :one, :typemin, :typemax))
    # Convenience default initializers.
    # First line covers both AbstractArray and ::Tuple{Vararg{AbstractArray, P}}
    @eval vmapreducethen(f, ::typeof($op), g, A::AbstractArray; dims=:, init=$init) = vmapreducethen(f, $op, g, init, A, dims)
    @eval vmapreducethen(f, ::typeof($op), g, As::Vararg{AbstractArray, P}; dims=:, init=$init) where {P} = vmapreducethen(f, $op, g, init, As, dims)
    # 3-argument versions for common binary ops
    @eval vmapreducethen(f::F, ::typeof($op), g::G, A::AbstractArray) where {F<:Function, G<:Function} = vmapreducethen(f, $op, g, $init, A, :)
    @eval vmapreducethen(f::F, ::typeof($op), g::G, As::Vararg{AbstractArray, P}) where {F<:Function, G<:Function, P} = vmapreducethen(f, $op, g, $init, As, :)
end


# mixed dimensions reduction

function staticdim_mapreducethen_quote(OP, I, static_dims::Vector{Int}, N::Int)
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
        # ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :𝒯ₒ))
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, Expr(:call, :f, A)))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, Expr(:call, :g, :ξ))
        push!(rblock.args, setb)
        return quote
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, T), Int)
            Bᵥ = $Bᵥ
            @turbo check_empty=true $loops
            return B
        end
    else
        # Pre-reduction
        # ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :𝒯ₒ))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, Expr(:call, :f, A)))
        push!(block.args, setξ)
        return quote
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, T), Int)
            Bᵥ = $Bᵥ
            $ξ
            @turbo check_empty=true $loops
            Bᵥ[] = g(ξ)
            return B
        end
    end
end

function branches_mapreducethen_quote(OP, I, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vmapreducethen!(f, op, g, init, B, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vmapreducethen!(f, op, g, init, B, A, $tc))))
            return q
        end
    end
    return staticdim_mapreducethen_quote(OP, I, static_dims, N)
end

@generated function _vmapreducethen!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, G, I, Tₒ, T, N, M, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_mapreducethen_quote(OP, I, N, M, D)
end

# this is the case of mapreducethen on a single array when rinds = ∅
function mapthen_quote()
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :A, :B)), block)
    setb = Expr(:(=), Expr(:ref, :B, :i), Expr(:call, :g, Expr(:call, :f, Expr(:ref, :A, :i))))
    push!(block.args, setb)
    return quote
        @turbo check_empty=true $loops
        return B
    end
end

@generated function _vmapreducethen!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, G, I, Tₒ, T, N}
    mapthen_quote()
end

################
# Version wherein an initial value is supplied

function vmapreducethen(f::F, op::OP, g::G, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, G, I<:Number, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(g, Base.promote_op(op, Base.promote_op(f, T), Int)), Dᴮ′)
    _vmapreducethen_init!(f, op, g, init, B, A, dims)
    return B
end

# reduction over all dims
@generated function vmapreducethen(f::F, op::OP, g::G, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, G, I<:Number, T, N}
    opsym = OP.instance
    quote
        ξ = convert(Base.promote_op($opsym, Base.promote_op(f, $T), Int), init)
        @turbo check_empty=true for i ∈ eachindex(A)
            ξ = $opsym(ξ, f(A[i]))
        end
        return g(ξ)
    end
end

function staticdim_mapreducethen_init_quote(OP, static_dims::Vector{Int}, N::Int)
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
        ξ₀ = Expr(:call, :convert, :𝒯ₒ, :init)
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, :ξ₀)
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, Expr(:call, :f, A)))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, Expr(:call, :g, :ξ))
        push!(rblock.args, setb)
        return quote
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, T), Int)
            Bᵥ = $Bᵥ
            ξ₀ = $ξ₀
            @turbo check_empty=true $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, :convert, :𝒯ₒ, :init))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, Expr(:call, :f, A)))
        push!(block.args, setξ)
        return quote
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, T), Int)
            Bᵥ = $Bᵥ
            $ξ
            @turbo check_empty=true $loops
            Bᵥ[] = g(ξ)
            return B
        end
    end
end

function branches_mapreducethen_init_quote(OP, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vmapreducethen_init!(f, op, g, init, B, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vmapreducethen_init!(f, op, g, init, B, A, $tc))))
            return q
        end
    end
    return staticdim_mapreducethen_init_quote(OP, static_dims, N)
end

@generated function _vmapreducethen_init!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, G, I, Tₒ, T, N, M, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_mapreducethen_init_quote(OP, N, M, D)
end
@generated function _vmapreducethen_init!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, G, I, Tₒ, T, N}
    mapthen_quote()
end

################
# Threaded version

"""
    vtmapreducethen(f, op, g, A::AbstractArray; dims=:, init)

Apply function `f` to each element of `A`, reduce the result over the dimensions
`dims` using the binary function `op`, then apply `g` to the result. Equivalent to
`g.(mapreduce(f, op, A, dims=dims, init=init))` but avoids the intermediate implied
by said expression while also fusing the post-transform `g` such that the output array
is populated in a single pass. Threaded.

The reduction necessitates an initial value `init` which may be `<:Number` or a function
which accepts a single type argument (e.g. `zero`);
`init` is optional for binary operators `+`, `*`, `min`, and `max`.

# Examples
```jldoctest
julia> vtmapreducethen(abs2, +, √, [1 2; 3 4], dims=1)    # L₂-norm; see `vnorm`
1×2 Matrix{Float64}:
 3.16228  4.47214

julia> vtmapreducethen(abs2, +, √, [1 2; 3 4], dims=2, init=1000.0)
2×1 Matrix{Float64}:
 31.701734968294716
 32.01562118716424

julia> vtmapreducethen(exp, +, log, [5 6; 7 8], dims=1)    # LSE, but recommend `vlogsumexp`
1×2 Matrix{Float64}:
 7.12693  8.12693
```
"""
function vtmapreducethen(f::F, op::OP, g::G, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, G, I, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(g, Base.promote_op(op, Base.promote_op(f, T), Int)), Dᴮ′)
    _vtmapreducethen!(f, op, g, init, B, A, dims)
    return B
end
vtmapreducethen(f, op, g, init, A, dims::Int) = vtmapreducethen(f, op, g, init, A, (dims,))

function vtmapreducethen!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}) where {F, OP, G, I, Tₒ, T, N}
    Dᴬ = size(A)
    Dᴮ = size(B)
    dims = Tuple((d for d ∈ eachindex(Dᴮ) if isone(Dᴮ[d])))
    all(d -> Dᴮ[d] == Dᴬ[d], (d for d ∈ eachindex(Dᴮ) if !isone(Dᴮ[d]))) || throw(DimensionMismatch)
    _vtmapreducethen!(f, op, g, init, B, A, dims)
    return B
end

@generated function vtmapreducethen(f::F, op::OP, g::G, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, G, I, T, N}
    opsym = OP.instance
    initsym = I.instance
    quote
        ξ = $initsym(Base.promote_op($opsym, Base.promote_op(f, $T), Int))
        @tturbo check_empty=true for i ∈ eachindex(A)
            ξ = $opsym(ξ, f(A[i]))
        end
        return g(ξ)
    end
end
vtmapreducethen(f::F, op::OP, g::G, init::I, A::AbstractArray) where {F, OP, G, I} = vtmapreducethen(f, op, g, init, A, :)

for (op, init) ∈ zip((:+, :*, :max,:min), (:zero, :one, :typemin, :typemax))
    # Convenience default initializers.
    # First line covers both AbstractArray and ::Tuple{Vararg{AbstractArray, P}}
    @eval vtmapreducethen(f, ::typeof($op), g, A::AbstractArray; dims=:, init=$init) = vtmapreducethen(f, $op, g, init, A, dims)
    @eval vtmapreducethen(f, ::typeof($op), g, As::Vararg{AbstractArray, P}; dims=:, init=$init) where {P} = vtmapreducethen(f, $op, g, init, As, dims)
    # 3-argument versions for common binary ops
    @eval vtmapreducethen(f::F, ::typeof($op), g::G, A::AbstractArray) where {F<:Function, G<:Function} = vtmapreducethen(f, $op, g, $init, A, :)
    @eval vtmapreducethen(f::F, ::typeof($op), g::G, As::Vararg{AbstractArray, P}) where {F<:Function, G<:Function, P} = vtmapreducethen(f, $op, g, $init, As, :)
end


# mixed dimensions reduction

function staticdim_tmapreducethen_quote(OP, I, static_dims::Vector{Int}, N::Int)
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
        # ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :𝒯ₒ))
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, Expr(:call, :f, A)))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, Expr(:call, :g, :ξ))
        push!(rblock.args, setb)
        return quote
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, T), Int)
            Bᵥ = $Bᵥ
            @tturbo check_empty=true $loops
            return B
        end
    else
        # Pre-reduction
        # ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), Expr(:call, :eltype, :Bᵥ)))
        ξ = Expr(:(=), :ξ, Expr(:call, Symbol(I.instance), :𝒯ₒ))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, Expr(:call, :f, A)))
        push!(block.args, setξ)
        return quote
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, T), Int)
            Bᵥ = $Bᵥ
            $ξ
            @tturbo check_empty=true $loops
            Bᵥ[] = g(ξ)
            return B
        end
    end
end

function branches_tmapreducethen_quote(OP, I, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtmapreducethen!(f, op, g, init, B, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vtmapreducethen!(f, op, g, init, B, A, $tc))))
            return q
        end
    end
    return staticdim_tmapreducethen_quote(OP, I, static_dims, N)
end

@generated function _vtmapreducethen!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, G, I, Tₒ, T, N, M, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_tmapreducethen_quote(OP, I, N, M, D)
end

# this is the case of mapreducethen on a single array when rinds = ∅
function tmapthen_quote()
    block = Expr(:block)
    loops = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :A, :B)), block)
    setb = Expr(:(=), Expr(:ref, :B, :i), Expr(:call, :g, Expr(:call, :f, Expr(:ref, :A, :i))))
    push!(block.args, setb)
    return quote
        @tturbo check_empty=true $loops
        return B
    end
end

@generated function _vtmapreducethen!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, G, I, Tₒ, T, N}
    tmapthen_quote()
end

################
# Version wherein an initial value is supplied

function vtmapreducethen(f::F, op::OP, g::G, init::I, A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {F, OP, G, I<:Number, T, N, M}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Base.promote_op(g, Base.promote_op(op, Base.promote_op(f, T), Int)), Dᴮ′)
    _vtmapreducethen_init!(f, op, g, init, B, A, dims)
    return B
end

# reduction over all dims
@generated function vtmapreducethen(f::F, op::OP, g::G, init::I, A::AbstractArray{T, N}, ::Colon) where {F, OP, G, I<:Number, T, N}
    opsym = OP.instance
    quote
        ξ = convert(Base.promote_op($opsym, Base.promote_op(f, $T), Int), init)
        @tturbo check_empty=true for i ∈ eachindex(A)
            ξ = $opsym(ξ, f(A[i]))
        end
        return g(ξ)
    end
end

function staticdim_tmapreducethen_init_quote(OP, static_dims::Vector{Int}, N::Int)
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
        ξ₀ = Expr(:call, :convert, :𝒯ₒ, :init)
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        ξ = Expr(:(=), :ξ, :ξ₀)
        push!(rblock.args, ξ)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, Expr(:call, :f, A)))
        push!(block.args, setξ)
        setb = Expr(:(=), Bᵥ′, Expr(:call, :g, :ξ))
        push!(rblock.args, setb)
        return quote
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, T), Int)
            Bᵥ = $Bᵥ
            ξ₀ = $ξ₀
            @tturbo check_empty=true $loops
            return B
        end
    else
        # Pre-reduction
        ξ = Expr(:(=), :ξ, Expr(:call, :convert, :𝒯ₒ, :init))
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = axes(A, $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = axes(A, $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setξ = Expr(:(=), :ξ, Expr(:call, Symbol(OP.instance), :ξ, Expr(:call, :f, A)))
        push!(block.args, setξ)
        return quote
            𝒯ₒ = Base.promote_op($(Symbol(OP.instance)), Base.promote_op(f, T), Int)
            Bᵥ = $Bᵥ
            $ξ
            @tturbo check_empty=true $loops
            Bᵥ[] = g(ξ)
            return B
        end
    end
end

function branches_tmapreducethen_init_quote(OP, N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtmapreducethen_init!(f, op, g, init, B, A, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vtmapreducethen_init!(f, op, g, init, B, A, $tc))))
            return q
        end
    end
    return staticdim_tmapreducethen_init_quote(OP, static_dims, N)
end

@generated function _vtmapreducethen_init!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::D) where {F, OP, G, I, Tₒ, T, N, M, D<:Tuple{Vararg{IntOrStaticInt, M}}}
    branches_tmapreducethen_init_quote(OP, N, M, D)
end
@generated function _vtmapreducethen_init!(f::F, op::OP, g::G, init::I, B::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, dims::Tuple{}) where {F, OP, G, I, Tₒ, T, N}
    tmapthen_quote()
end
