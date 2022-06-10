#
# Date created: 2022-03-17
# Author: aradclif
#
#
############################################################################################
# Assorted performance tests
using BenchmarkTools
A = rand(10,10,10);

ns = [1, 5, 10, 50]
bs = Matrix{NTuple{3, BenchmarkTools.Trial}}(undef, length(ns), 4);
for d = 1:4
    for (i, n) ∈ enumerate(ns)
        A = rand(n, 2n, 3n, 4n);
        bs[i, d] = ((@benchmark findmax(A, dims=($d,))),
                    (@benchmark vfindmax2(A, ($d,))),
                    (@benchmark vfindmax3(A, ($d,))))
    end
end

@benchmark extrema(A, dims=(1,3))
@benchmark lvextrema(A, dims=(1,3), multithreaded=false)
@benchmark maximum(A, dims=(1,3))
@benchmark lvmaximum(A, dims=(1,3), multithreaded=false)
maximum(A, dims=(1,3)) == lvmaximum(A, dims=(1,3), multithreaded=false)

@benchmark lvsum(A, dims=(1,2), multithreaded=false)
@benchmark lvsum(identity, A, dims=(1,2))
@benchmark lvsum(A, dims=:, multithreaded=false)
@benchmark lvsum(identity, A, dims=:)
@benchmark lvsum(A)
@benchmark lvsum(identity, A)

@code_warntype lvreduce(+, A)
@code_warntype _lvreduce(+, A)

A = rand(10, 10, 10, 10000);
dims = (2,4)
dims = (3,4)
A = rand(10, 10000, 10, 10);
dims = (1,2)
dims = (1,2,3)
dims = (2,3,4)
dims = (1,3,4)
@benchmark sum(A)
@benchmark lvsum(A)
@benchmark mean(A)
@benchmark lvmean(A)
@benchmark var(A)
@benchmark lvvar(A)
@benchmark std(A)
@benchmark lvstd(A)
@benchmark sum(A, dims=dims)
@benchmark lvsum(A, dims=dims)
@benchmark mean(A, dims=dims)
@benchmark lvmean(A, dims=dims)
@benchmark var(A, dims=dims)
@benchmark lvvar(A, dims=dims)
@benchmark std(A, dims=dims)
@benchmark lvstd(A, dims=dims)
@benchmark lvmaximum(A, dims=dims)

@benchmark vsum(A)
@benchmark vmean(A)
@benchmark vvar(A)
@benchmark vstd(A)
@benchmark vsum(A, dims=dims)
@benchmark vmean(A, dims=dims)
@benchmark vvar(A, dims=dims)
@benchmark vstd(A, dims=dims)
@benchmark vmaximum(A, dims=dims)

@benchmark lvsum(A, dims=dims)
@benchmark vsum(A, dims=dims)
@benchmark lvsum3(A, dims=dims)
@benchmark _lvreduce3(+, zero, A, dims)
@benchmark lvsum4(A, dims=dims)
@benchmark _lvsum3(A, dims)
@benchmark myvsum(A, dims)
@benchmark myvsum2(A, dims)
@benchmark VectorizedStatistics._vsum(A, dims)
@benchmark lvsum3(identity, A, dims=dims)
function myvsum(A::AbstractArray{T, N}, dims) where {T, N}
    Dᴬ = size(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))
    B = similar(A, Dᴮ′)
    VectorizedStatistics._vsum!(B, A, dims)
end

lvsum5(A, dims) = _lvreduce3(+, zero, A, dims)
@benchmark lvsum5(A, dims)

@benchmark _lvreduce3(*, one, A, dims)
@benchmark vprod(A, dims=dims)
@benchmark lvprod(A, dims=dims)

@benchmark lvmaximum(A, dims=dims)
@benchmark lvmaximum3(A, dims=dims)
@benchmark _lvreduce3(max, typemin, A, dims)
@benchmark vmaximum(A, dims=dims)

@benchmark lvsum(abs2, A, dims=dims)
@benchmark _lvmapreduce3(abs2, +, zero, A, dims)
@benchmark lvsum(cos, A, dims=dims)
@benchmark _lvmapreduce3(cos, +, zero, A, dims)
@benchmark lvsum(exp, A, dims=dims)
@benchmark _lvmapreduce3(exp, +, zero, A, dims)
@benchmark lvmaximum(exp, A, dims=dims)
@benchmark _lvmapreduce3(exp, max, typemin, A, dims)
_lvmapreduce3(exp, max, typemin, A, dims) ≈ maximum(exp, A, dims=dims)

@benchmark lvmean(A, dims=dims)
@benchmark vmean(A, dims=dims)
@benchmark _lvmean3(A, dims)

@benchmark lvsum(abs2, A, dims=dims)
@benchmark sum(abs2, A, dims=dims)
@benchmark _lvmapreduce3(abs2, +, zero, A, dims)
@benchmark vvmapreduce(abs2, +, zero, A, dims)
@benchmark vvsum(abs2, A, dims)
myf(x) = x * x
myf2(x) = x * x * x * x
@benchmark vvsum(myf, A, dims)
@benchmark vvsum(myf2, A, dims)

@benchmark sum(A, dims=dims)
@benchmark vvsum(A, dims)

#
@benchmark vextrema2(A, dims)
@benchmark extrema(A, dims=dims)
@benchmark vextrema(A, dims=dims)
@benchmark lvextrema(A, dims=dims)

@benchmark vvextrema(A, dims)
x = randn(1000);
@benchmark vvextrema(x, (1,))
@benchmark extrema(x)

@benchmark vvmaximum(A, dims)
@benchmark vvminimum(A, dims)

@benchmark sum(A)
@benchmark vvsum(A)
@benchmark VectorizedStatistics._vsum(A, :)
@timev B = VectorizedStatistics._vsum(A, dims)
@code_warntype VectorizedStatistics._vsum!(B,A, dims)

@benchmark vvlogsumexp(A, (2,3,4))
@benchmark vvlogsumexp2(A, (2,3,4))
@benchmark vvlogsumexp(A, (1,3))
@benchmark vvlogsumexp2(A, (1,3))
@benchmark vvlogsumexp(A, (2,4))
@benchmark vvlogsumexp2(A, (2,4))
@benchmark vvlogsumexp(A, (2,))
@benchmark vvlogsumexp2(A, (2,))

A = rand(5,5,5,5);

@benchmark vvlogsoftmax(A, (2,3,4))
@benchmark _lvlogsoftmax(A, (2,3,4))
@benchmark vvlogsoftmax(A, (1,3))
@benchmark _lvlogsoftmax(A, (1,3))
@benchmark vvlogsoftmax(A, (2,4))
@benchmark _lvlogsoftmax(A, (2,4))
@benchmark vvlogsoftmax(A, (2,))
@benchmark _lvlogsoftmax(A, (2,))
vvlogsoftmax(A, (2,3,4)) ≈ _lvlogsoftmax(A, (2,3,4))
vvlogsoftmax(A, (1,3)) ≈ _lvlogsoftmax(A, (1,3))
vvlogsoftmax(A, (2,4)) ≈ _lvlogsoftmax(A, (2,4))
vvlogsoftmax(A, (2,)) ≈ _lvlogsoftmax(A, (2,))
@benchmark vvlogsoftmax(A)


################
@benchmark findmax(A, dims=(2,3,4))
@benchmark vvfindminmax(identity, >, typemin, A, (1, 2,3,4))
@benchmark vvfindminmax(abs2, >, typemin, A, (2,3,4))
vvfindminmax(identity, >, typemin, A, (1,3))

W = rand(ntuple(_ -> 5, 9)...);
@timev vvmapreduce(identity, +, zero, W, (1,3,5,7,9,10));
@timev vvmapreduce(identity, +, zero, W, (1,2,3,4,5,6));

################
# time/cost of type inference in vararg method
# Conclusion: seems to be little difference between the methods. If any difference,
# it is barely measurable at tenths of a nanosecond.

function option1(f::F, op::OP, As::Tuple{Vararg{S, P}}) where {F, OP, T, N, S<:AbstractArray{T, N}, P}
    Base.promote_op(op, Base.promote_op(f, ntuple(_ -> T, Val(P))...), Int)
end

function option2(f::F, op::OP, As::Tuple{Vararg{AbstractArray, P}}) where {F, OP, P}
    Base.promote_op(op, Base.promote_op(f, ntuple(p -> eltype(As[p]), Val(P))...), Int)
end
as3 = (A1, A2, A3);
as4 = (A1, A2, A3, A4);
as4_2 = (A1, A2, A3, A6);
as8 = (A1, A2, A3, A4, A5, A5, A5, A5);
fg(x, y, z) = x*y + z
fg2(x, y, z) = x*y + exp(z)
fg4(x, y, z, w) = x*y + exp(z) / w
fg4_b(x, y, z, w) = ≥(x*y + exp(z) / w, 1)
fg8(x, y, z, w, u1, u2, u3, u4) = x*y + exp(z) / w + u1*u2 / u3^u4
@benchmark option1(fg2, +, as3)
@benchmark option2(fg2, +, as3)
@benchmark option1($fg4_b, $+, $as4_2)
@benchmark option2($fg4_b, $+, $as4_2)
@benchmark option1($fg8, $+, $as8)
@benchmark option2($fg8, $+, $as8)

# time/cost of forming the new dimensions
# Conclusin: seems to be little difference between the methods.
function newdims1(As::Tuple{Vararg{S, P}}, dims::NTuple{M, Int}) where {T, N, S<:AbstractArray{T, N}, P, M}
    ax = axes(As[1])
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), Val(N))
end

function newdims2(As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {P, M}
    ax = axes(As[1])
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : length(ax[d]), ndims(As[1]))
end

function newdims4(As::Tuple{Vararg{AbstractArray, P}}, dims::NTuple{M, Int}) where {P, M}
    A = As[1]
    Dᴬ = size(A)
    ax = axes(A)
    Dᴮ′ = ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], ndims(As[1]))
end

dims1 = (2,)
dims2 = (2,4)
dims3 = (2,3,4)
@benchmark newdims1(as3, dims1)
@benchmark newdims2(as3, dims1)
@benchmark newdims1(as3, dims2)
@benchmark newdims2(as3, dims2)
@benchmark newdims1($as3, $dims3)
@benchmark newdims2($as3, $dims3)
@benchmark newdims4($as3, $dims3)
@benchmark newdims1(as4_2, dims3)
@benchmark newdims2(as4_2, dims3)
@benchmark newdims4($as4_2, $dims3)

################
A1 = rand(5,5,5,5);
A2 = rand(5,5,5,5);
A3 = rand(5,5,5,5);
A4 = rand(5,5,5,5);
A5 = rand(5,5,5,5);
A6 = rand(1:10, 5,5,5,5);
as = (A1, A2, A3);
################
# Simple tests
@benchmark mapreduce(abs2, +, A1, dims=(1,2,4))
@benchmark vvmapreduce(abs2, +, A1, dims=(1,2,4))
@benchmark prod(A1, dims=1)
@benchmark vvprod(A1, dims=1)
@benchmark extrema(log, A1, dims=(1,2))
@benchmark vvextrema(log, A1, dims=(1,2))
@benchmark extrema(A1, dims=(1,2))
@benchmark vvextrema(A1, dims=(1,2))

# In README
@benchmark mapreduce($abs2, $+, $A1, dims=$(1,2,4))
@benchmark vvmapreduce($abs2, $+, $A1, dims=$(1,2,4))
@benchmark extrema($A1, dims=$(1,2))
@benchmark vvextrema($A1, dims=$(1,2))
@benchmark extrema($A1, dims=$(3,4))
@benchmark vvextrema($A1, dims=$(3,4))

################
# Tests and example use for varargs mapreduce
@benchmark vvmapreduce(+, +, zero, as, (1,2,4))
@benchmark vvmapreduce(+, +, as, dims=(1,2,4), init=zero)
@benchmark mapreduce(+, +, A1, A2, A3, dims=(1,2,4))
@benchmark vvmapreduce(+, +, A1, A2, A3, dims=(1,2,4))
@tullio out[1, 1, i_3, 1] := A1[i_1, i_2, i_3, i_4] + A2[i_1, i_2, i_3, i_4] + A3[i_1, i_2, i_3, i_4]
@benchmark mapreduce($+, $+, $A1, $A2, $A3, $A4, dims=$(1,2,4))
@benchmark vvmapreduce($+, $+, $A1, $A2, $A3, $A4, dims=$(1,2,4))
h(w, x, y, z) = w * x + y * z
@benchmark mapreduce($h, $+, $A1, $A2, $A3, $A4, dims=$(1,2,4))
@benchmark vvmapreduce($h, $+, $A1, $A2, $A3, $A4, dims=$(1,2,4))
vvmapreduce(+, +, zero, as, (1,2,4)) ≈ mapreduce(+, +, A1, A2, A3, dims=(1, 2,4))
g(x, y, z) = x * y + z
@benchmark vvmapreduce(g, +, zero, as, (1,2,4))
vvmapreduce((x, y, z) -> x+y+z, +, zero, as, (1,2,3,4))
vvmapreduce(+, +, zero, as, (5,)) ≈ mapreduce(+, +, A1, A2, A3, dims=5)
@benchmark vvmapreduce(+, +, zero, as, (5,))
@benchmark vmap(+, as...)
@benchmark vvmapreduce(+, +, as..., dims=5)

# Tests of variably typed arrays
A4 = rand(1:5, 5,5,5,5);
@benchmark vvmapreduce(+, +, zero, (A1, A2), (2,3,4))
@benchmark vvmapreduce(+, +, zero, (A1, A4), (2,3,4))
vvmapreduce(+, +, zero, (A1, A4), (2,3,4)) ≈ mapreduce(+, +, A1, A4, dims=(2,3,4))
@benchmark vvmapreduce(+, +, (A1, A4), dims=(2,3,4))

# A rather absurd performance difference
@benchmark vvmapreduce((x,y,z,w) -> x*y*z*w, +, zero, 1:10, 11:20, 21:30, 31:40)
@benchmark mapreduce((x,y,z,w) -> x*y*z*w, +, 1:10, 11:20, 21:30, 31:40)
@benchmark vvmapreduce((x,y,z,w,u) -> x*y*z*w*u, +, zero, 1:10, 11:20, 21:30, 31:40, 41:50)
@benchmark mapreduce((x,y,z,w,u) -> x*y*z*w*u, +, 1:10, 11:20, 21:30, 31:40, 41:50)
@benchmark vvmapreduce((x,y,z,w) -> x*y*z*w, +, 1:10, 11:20, 21:30, 31:40)
@benchmark vvmapreduce((x,y,z,w) -> x*y*z*w, +, (1:10, 11:20, 21:30, 31:40))
@benchmark vvmapreduce((x,y,z,w) -> x*y*z*w, +, 1.:10, 11:20, 21:30, 31:40)
@benchmark vvmapreduce((x,y,z,w) -> x*y*z*w, +, 1.:10, 11.:20, 21.:30, 31.:40)
@benchmark vvmapreduce(abs2, +, 1:10)
@benchmark mapreduce(abs2, +, 1:10)

@benchmark mapreduce($h, $+, $1:10, $11:20, $21:30, $31:40)
@benchmark vvmapreduce($h, $+, $1:10, $11:20, $21:30, $31:40)
@benchmark mapreduce($h, $+, $1:100, $101:200, $201:300, $301:400)
@benchmark vvmapreduce($h, $+, $1:100, $101:200, $201:300, $301:400)
@benchmark mapreduce($h, $+, $1:1000, $1001:2000, $2001:3000, $3001:4000)
@benchmark vvmapreduce($h, $+, $1:1000, $1001:2000, $2001:3000, $3001:4000)

# interface tests
@benchmark vvmapreduce(*, +, zero, A1, A2, A3)
@benchmark vvmapreduce(*, +, A1, A2, A3)
@benchmark vvmapreduce(*, +, A1, A2, A3, dims=:)
@benchmark vvmapreduce(*, +, A1, A2, A3, dims=:, init=0)
@benchmark vvmapreduce(*, +, A1, A2, A3, dims=:, init=zero)
@benchmark vvmapreduce(*, +, as)

# Notably, if ≥ 4 slurped array args, then * slows down and allocates a lot for reasons unknown.
# oddly, if one manually writes the operations, then the cost is as it should be.
@benchmark vvmapreduce(*, +, A1, A2, A3, A4)
@benchmark vvmapreduce((x,y,z,w) -> x*y*z*w, +, zero, A1, A2, A3, A4)
@benchmark vvmapreduce((x,y,z,w) -> x*y*z*w, +, A1, A2, A3, A4, dims=:, init=zero)
@benchmark vvmapreduce(+, +, A1, A2, A3, A4)

# And for really strange stuff (e.g. posterior predictive transformations)
@benchmark vvmapreduce((x,y,z) -> ifelse(x*y+z ≥ 1, 1, 0), +, $A1, $A2, $A3)
@benchmark vvmapreduce((x,y,z) -> ifelse(x*y+z ≥ 1, 1, 0), +, A1, A2, A3, dims=(2,3,4))
# using ifelse for just a boolean is quite slow, but the above is just for demonstration
@benchmark vvmapreduce(≥, +, A1, A2)
@benchmark vvmapreduce((x,y,z) -> ≥(x*y+z, 1), +, $A1, $A2, $A3)
@benchmark vvmapreduce((x,y,z) -> ≥(x*y+z, 1), +, A1, A2, A3, dims=(2,3,4))
@benchmark mapreduce((x,y,z) -> ≥(x*y+z, 1), +, A1, A2, A3)
# What I mean by posterior predictive transformation? Well, one might encounter
# this in Bayesian model checking, which provides a convenient example.
# If one wishes to compute the Pr = ∫∫𝕀(T(yʳᵉᵖ, θ) ≥ T(y, θ))p(yʳᵉᵖ|θ)p(θ|y)dyʳᵉᵖdθ
# Let's imagine that A1 represents T(yʳᵉᵖ, θ) and A2 represents T(y, θ)
# i.e. the test variable samples computed as a functional of the Markov chain (samples of θ)
# Then, Pr is computed as
vvmapreduce(≥, +, A1, A2) / length(A1)
# Or, if only the probability is of interest, and we do not wish to use the functionals
# for any other purpose, we could compute it as:
vvmapreduce((x, y) -> ≥(f(x), f(y)), +, A1, A2)
mapreduce((x, y) -> ≥(f(x), f(y)), +, A1, A2)
# where `f` is the functional of interest, e.g.
@benchmark vvmapreduce((x, y) -> ≥(abs2(x), abs2(y)), +, A1, A2)
@benchmark vvmapreduce((x, y) -> ≥(abs2(x), abs2(y)), +, A1, A2, dims=(2,3,4))

# One can also express commonly encountered reductions with ease;
# these will be fused once a post-reduction operator can be specified
# MSE
@benchmark vvmapreduce((x, y) -> abs2(x - y), +, A1, A2, dims=(2,4)) ./ (size(A1, 2) * size(A1, 4) )
@benchmark mapreduce((x, y) -> abs2(x - y), +, A1, A2, dims=(2,4)) ./ (size(A1, 2) * size(A1, 4) )
# Euclidean distance
B = (√).(vvmapreduce((x, y) -> abs2(x - y), +, A1, A2, dims=(2,4)))
@test B ≈ (√).(mapreduce((x, y) -> abs2(x - y), +, A1, A2, dims=(2,4)))

# multi-threading examples
B1 = rand(20,20,20,20);
B2 = rand(20,20,20,20);
B3 = rand(20,20,20,20);
B4 = rand(20,20,20,20);
@benchmark vvmapreduce(*, +, zero, B1, B2, B3)
@benchmark vvmapreduce(*, +, B1, B2, B3)
@benchmark vvmapreduce(*, +, B1, B2, B3, dims=:)
@benchmark vvmapreduce(*, +, B1, B2, B3, dims=:, init=0)
@benchmark vvmapreduce(*, +, B1, B2, B3, dims=:, init=zero)
@benchmark vtmapreduce(*, +, zero, B1, B2, B3)
@benchmark vtmapreduce(*, +, B1, B2, B3)
@benchmark vtmapreduce(*, +, B1, B2, B3, dims=:)
@benchmark vtmapreduce(*, +, B1, B2, B3, dims=:, init=0)
@benchmark vtmapreduce(*, +, B1, B2, B3, dims=:, init=zero)

@benchmark vvmapreduce(*, +, B1, B2, B3, dims=(2,3,4))
@benchmark vtmapreduce(*, +, B1, B2, B3, dims=(2,3,4))
@benchmark vvmapreduce(*, +, B1, B2, B3, dims=(2,4))
@benchmark vtmapreduce(*, +, B1, B2, B3, dims=(2,4))
@benchmark vvmapreduce(*, +, B1, B2, B3, dims=(1,3))
@benchmark vtmapreduce(*, +, B1, B2, B3, dims=(1,3))
@benchmark vvmapreduce(*, +, B1, B2, B3, dims=(1,))
@benchmark vtmapreduce(*, +, B1, B2, B3, dims=(1,))

################
# Tests and example usage for vvmap, vtmap: as expected, slower than
# the highly optimized versions provided in LoopVectorization.jl
@benchmark vmap((x, y, z) -> x + y * z, A1, A2, A3)
@benchmark vvmap((x, y, z) -> x + y * z, A1, A2, A3)
h(x, y, z) = x + y * z
@benchmark vmap(h, A1, A2, A3)
@benchmark vvmap(h, A1, A2, A3)

################
# Tests of varargs findmin, findmax

#### tests
# over all
A1 = rand(5,5);
A2 = rand(5,5);
A3 = rand(5,5);
as = (A1, A2, A3);
@benchmark vfindminmax(+, >, typemin, as, :)
A′ = @. A1 + A2 + A3;
findmax(A′)
vfindmax(+, A1, A2, A3)
vfindmax(+, as)

v1 = rand(5);
v2 = rand(5);
v3 = rand(5);
vs = (v1, v2, v3);
@benchmark vfindminmax(+, >, typemin, vs, :)
v′ = @. v1 + v2 + v3;
findmax(v′)

# on subset of dims
vfindminmax(+, >, typemin, as, (2,)) == findmax(A′, dims=2)
@benchmark vfindminmax(+, >, typemin, as, (2,))
@benchmark findmax(A′, dims=2)
vfindmax(+, A1, A2, A3, dims=2)
vfindmax(+, as, dims=2)
vfindmax(+, as, 2)
vfindmax(as)

# anonymous functions
vfindmax((x, y, z) -> x * y + z, A1, A2, A3)
A′ = @. A1 * A2 + A3;
findmax(A′)

# light performance tests
B1 = rand(5,5,5,5);
B2 = rand(5,5,5,5);
B3 = rand(5,5,5,5);
bs = (B1, B2, B3);
@benchmark vfindminmax(+, >, typemin, bs, :)
B′ = @. B1 + B2 + B3;
findmin(B′) == vfindmin(+, B1, B2, B3)
@benchmark findmin(@. $B1 + $B2 + $B3)
@benchmark vfindmin(+, $B1, $B2, $B3)

@benchmark findmin((@. $B1 + $B2 + $B3), dims=(2,4))
@benchmark vfindmin(+, $B1, $B2, $B3, dims=(2,4))

@benchmark findmin((@. abs2($B1) * $B2 + $B3), dims=$(3,4))
@benchmark vfindmin((x, y, z) -> abs2(x) * y + z, $B1, $B2, $B3, dims=$(3,4))

@benchmark vfindmax(+, bs)
@benchmark vfindmax((x, y, z) -> x * y + z, B1, B2, B3)


# light performance tests
C1 = rand(50,50,50,50);
C2 = rand(50,50,50,50);
C3 = rand(50,50,50,50);
cs = (C1, C2, C3);
@benchmark vfindminmax(+, >, typemin, cs, :)
C′ = @. C1 + C2 + C3;
findmax(C′)
@benchmark vfindmax(+, C1, C2, C3)
@benchmark vfindmax(+, cs)
@benchmark vfindmax((x, y, z) -> x * y + z, C1, C2, C3)
@benchmark vtfindmax(+, C1, C2, C3)
@benchmark vtfindmax(+, cs)
@benchmark vtfindmax((x, y, z) -> x * y + z, C1, C2, C3)

@benchmark findmin(@. $C1 + $C2 + $C3)
@benchmark vfindmin(+, $C1, $C2, $C3)
@benchmark vtfindmin(+, $C1, $C2, $C3)

@benchmark findmin((@. $C1 + $C2 + $C3), dims=(2,4))
@benchmark vfindmin(+, $C1, $C2, $C3, dims=(2,4))
@benchmark vtfindmin(+, $C1, $C2, $C3, dims=(2,4))

@benchmark findmin((@. abs2($C1) * $C2 + $C3), dims=$(3,4))
@benchmark vfindmin((x, y, z) -> abs2(x) * y + z, $C1, $C2, $C3, dims=$(3,4))
@benchmark vtfindmin((x, y, z) -> abs2(x) * y + z, $C1, $C2, $C3, dims=$(3,4))

################
# Comparison of vextrema: with/without zip
A = rand(3,3,3,3);
A = rand(5,5,5,5);
A = rand(10,10,10,10);
A = rand(3,3,3,3,3);
A = rand(5,5,5,5,5);

# non-zip is faster only when the array is very small (length(A) < 100),
# and/or when the reduction is taking place across a large chunk of the array.
# If the dimensions are equal size, as in the examples above, this typically requires
# > ndims(A) ÷ 2.
# As one would expect, there is dependence on memory traversal order, i.e.
# which dimensions are being reduced over, as they must appear in the innermost loop.
# This is easier to elicit with equal size dimensions. For unequal size dimensions,
# cost modeling is needed to determine what the optimal action -- out of scope for
# this little note.
# In any case, making all the loops available to LoopVectorization will yield
# superior performance in most cases, despite the need to zip the result.

# An aside: When the reduction occurs over all dimensions, there is clearly no penalty
# to the non-zip method. It is an unfortunate side effect, but it goes un-used in such
# a case.

tups = [(1,), (2,), (1,3), (2,4), (1,2,4), (2,3,4), (1,2,3,4)]
tup = tups[end]

@benchmark vextrema(identity, typemax, typemin, $A, $tup)
@benchmark vextrema_nonzip(identity, typemax, typemin, $A, $tup)
@benchmark vvextrema($A, $tup)
@benchmark extrema($A, dims=$tup)

for tup ∈ tups
    println("dims = ", tup, '\t', "size(A) = ", size(A))
    @btime vextrema(identity, typemax, typemin, $A, $tup)
    @btime vextrema_nonzip(identity, typemax, typemin, $A, $tup)
    @btime vvextrema($A, $tup)
    @btime extrema($A, dims=$tup)
end
