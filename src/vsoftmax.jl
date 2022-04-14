#
# Date created: 2022-03-23
# Author: aradclif
#
#
############################################################################################

# reduction over all dims
function vsoftmax(A::AbstractArray{T, N}, ::Colon) where {T, N}
    b = vlogsumexp(A)
    C = similar(A, Base.promote_op(exp, T))
    @turbo for i ∈ eachindex(A)
        C[i] = exp(A[i] - b)
    end
    C
end
# ::AbstractArray required in order for kwargs interface to work
"""
    vsoftmax(A::AbstractArray)

Compute the softmax function, treating the entire array as a single vector.
Care is taken to ensure that the computation will not overflow/underflow, but the caller
should be aware that `+Inf` and `NaN` are not handled.

See also: [`vlogsoftmax`](@ref)
"""
vsoftmax(A::AbstractArray) = vsoftmax(A, :)

"""
    vsoftmax(A::AbstractArray, dims)

Compute the softmax function, treating each slice of `A` specified by `dims` as if
it were a single vector; `dims` may be `::Int`, `::NTuple{M, Int} where {M}` or `::Colon`.
Avoids overflow/underflow, but `+Inf` and `NaN` are not handled.
"""
function vsoftmax(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    Dᴬ = size(A)
    B = vlogsumexp(A, dims)
    C = similar(A, Base.promote_op(exp, T))
    _vsoftmax!(C, A, B, dims)
    return C
end
vsoftmax(A, dims::Int) = vsoftmax(A, (dims,))

# Provide inherently inefficient kwargs interface. Requires ::AbstractArray in the locations
# indicated above.
"""
    vsoftmax(A::AbstractArray; dims=:)

Identical to non-keyword args version; slightly less performant due to use of kwargs.
"""
vsoftmax(A; dims=:) = vsoftmax(A, dims)

function staticdim_softmax_quote(static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    C = Expr(:ref, :C, ntuple(d -> Symbol(:i_, d), N)...)
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
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B, C), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B, C), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        lse = Expr(:(=), :lse, Bᵥ′)
        push!(rblock.args, lse)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, C), $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setc = Expr(:(=), C, Expr(:call, :exp, Expr(:call, :-, A, :lse)))
        push!(block.args, setc)
        return quote
            Bᵥ = $Bᵥ
            @turbo $loops
            return C
        end
    else
        # Pre-reduction
        lse = Expr(:(=), :lse, Bᵥ′)
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = indices((A, C), $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, C), $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setc = Expr(:(=), C, Expr(:call, :exp, Expr(:call, :-, A, :lse)))
        push!(block.args, setc)
        return quote
            Bᵥ = $Bᵥ
            $lse
            @turbo $loops
            return C
        end
    end
end

function branches_softmax_quote(N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vsoftmax!(C, A, B, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vsoftmax!(C, A, B, $tc))))
            return q
        end
    end
    return staticdim_softmax_quote(static_dims, N)
end

@generated function _vsoftmax!(C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, B::AbstractArray{Tₘ, N}, dims::D) where {Tₒ, T, Tₘ, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_softmax_quote(N, M, D)
end
@generated function _vsoftmax!(C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, B::AbstractArray{Tₘ, N}, dims::Tuple{}) where {Tₒ, T, Tₘ, N}
    :(copyto!(C, A); return C)
end

############################################################################################
# reduction over all dims
function vtsoftmax(A::AbstractArray{T, N}, ::Colon) where {T, N}
    b = vtlogsumexp(A)
    C = similar(A, Base.promote_op(exp, T))
    @tturbo for i ∈ eachindex(A)
        C[i] = exp(A[i] - b)
    end
    C
end
# ::AbstractArray required in order for kwargs interface to work
"""
    vtsoftmax(A::AbstractArray)

Compute the softmax function, treating the entire array as a single vector. Threaded.
Care is taken to ensure that the computation will not overflow/underflow, but the caller
should be aware that `+Inf` and `NaN` are not handled.

See also: [`vtlogsoftmax`](@ref)
"""
vtsoftmax(A::AbstractArray) = vtsoftmax(A, :)

"""
    vtsoftmax(A::AbstractArray, dims)

Compute the softmax function, treating each slice of `A` specified by `dims` as if
it were a single vector; `dims` may be `::Int`, `::NTuple{M, Int} where {M}` or `::Colon`.
Threaded. Avoids overflow/underflow, but `+Inf` and `NaN` are not handled.
"""
function vtsoftmax(A::AbstractArray{T, N}, dims::NTuple{M, Int}) where {T, N, M}
    Dᴬ = size(A)
    B = vtlogsumexp(A, dims)
    C = similar(A, Base.promote_op(exp, T))
    _vtsoftmax!(C, A, B, dims)
    return C
end
"""
    vtsoftmax(A::AbstractArray; dims=:)

Identical to non-keyword args version; slightly less performant due to use of kwargs. Threaded.
"""
vtsoftmax(A, dims::Int) = vtsoftmax(A, (dims,))

# Provide inherently inefficient kwargs interface. Requires ::AbstractArray in the locations
# indicated above.
vtsoftmax(A; dims=:) = vtsoftmax(A, dims)

function staticdim_tsoftmax_quote(static_dims::Vector{Int}, N::Int)
    A = Expr(:ref, :A, ntuple(d -> Symbol(:i_, d), N)...)
    Bᵥ = Expr(:call, :view, :B)
    Bᵥ′ = Expr(:ref, :Bᵥ)
    C = Expr(:ref, :C, ntuple(d -> Symbol(:i_, d), N)...)
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
        loops = Expr(:for, :($(Symbol(:i_, nrinds[1])) = indices((A, B, C), $(nrinds[1]))), block)
        for d ∈ @view(nrinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, B, C), $d)), newblock))
            block = newblock
        end
        rblock = block
        # Pre-reduction
        lse = Expr(:(=), :lse, Bᵥ′)
        push!(rblock.args, lse)
        # Reduction loop
        for d ∈ rinds
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, C), $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setc = Expr(:(=), C, Expr(:call, :exp, Expr(:call, :-, A, :lse)))
        push!(block.args, setc)
        return quote
            Bᵥ = $Bᵥ
            @tturbo $loops
            return C
        end
    else
        # Pre-reduction
        lse = Expr(:(=), :lse, Bᵥ′)
        # Reduction loop
        block = Expr(:block)
        loops = Expr(:for, :($(Symbol(:i_, rinds[1])) = indices((A, C), $(rinds[1]))), block)
        for d ∈ @view(rinds[2:end])
            newblock = Expr(:block)
            push!(block.args, Expr(:for, :($(Symbol(:i_, d)) = indices((A, C), $d)), newblock))
            block = newblock
        end
        # Push to inside innermost loop
        setc = Expr(:(=), C, Expr(:call, :exp, Expr(:call, :-, A, :lse)))
        push!(block.args, setc)
        return quote
            Bᵥ = $Bᵥ
            $lse
            @tturbo $loops
            return C
        end
    end
end

function branches_tsoftmax_quote(N::Int, M::Int, D)
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
                qnew = Expr(ifsym, :(dimm == $n), :(return _vtsoftmax!(C, A, B, $tc)))
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
            push!(qold.args, Expr(:block, :(return _vtsoftmax!(C, A, B, $tc))))
            return q
        end
    end
    return staticdim_tsoftmax_quote(static_dims, N)
end

@generated function _vtsoftmax!(C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, B::AbstractArray{Tₘ, N}, dims::D) where {Tₒ, T, Tₘ, N, M, D<:Tuple{Vararg{Integer, M}}}
    branches_tsoftmax_quote(N, M, D)
end
@generated function _vtsoftmax!(C::AbstractArray{Tₒ, N}, A::AbstractArray{T, N}, B::AbstractArray{Tₘ, N}, dims::Tuple{}) where {Tₒ, T, Tₘ, N}
    :(copyto!(C, A); return C)
end
