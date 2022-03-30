#
# Date created: 2022-03-23
# Author: aradclif
#
#
############################################################################################
A = rand(5,5,5,5);
@timev vvmapreduce(identity, min, typemin, A, (1,2,3));
@timev vvmapreduce(identity, min, typemin, A, (1,3,2));
@timev vvmapreduce(identity, min, typemin, A, (2,1,3));
@timev vvmapreduce(identity, min, typemin, A, (2,3,1));
@timev vvmapreduce(identity, min, typemin, A, (3,1,2));
@timev vvmapreduce(identity, min, typemin, A, (3,2,1));
#
@timev vvmapreduce(identity, min, typemin, A, (1,2,4));
@timev vvmapreduce(identity, min, typemin, A, (1,4,2));
@timev vvmapreduce(identity, min, typemin, A, (2,1,4));
@timev vvmapreduce(identity, min, typemin, A, (2,4,1));
@timev vvmapreduce(identity, min, typemin, A, (4,1,2));
@timev vvmapreduce(identity, min, typemin, A, (4,2,1));
#
@timev vvmapreduce(identity, min, typemin, A, (1,4,3));
@timev vvmapreduce(identity, min, typemin, A, (1,3,4));
@timev vvmapreduce(identity, min, typemin, A, (4,1,3));
@timev vvmapreduce(identity, min, typemin, A, (4,3,1));
@timev vvmapreduce(identity, min, typemin, A, (3,1,4));
@timev vvmapreduce(identity, min, typemin, A, (3,4,1));
#
@timev vvmapreduce(identity, min, typemin, A, (4,2,3));
@timev vvmapreduce(identity, min, typemin, A, (4,3,2));
@timev vvmapreduce(identity, min, typemin, A, (2,4,3));
@timev vvmapreduce(identity, min, typemin, A, (2,3,4));
@timev vvmapreduce(identity, min, typemin, A, (3,4,2));
@timev vvmapreduce(identity, min, typemin, A, (3,2,4));
#
@timev minimum(A, dims=(1,2,3));
@timev minimum(A, dims=(1,3,2));
@timev minimum(A, dims=(2,1,3));
@timev minimum(A, dims=(2,3,1));
@timev minimum(A, dims=(3,1,2));
@timev minimum(A, dims=(3,2,1));

vvmapreduce(identity, min, typemin, A, (1,2,3)) == vvmapreduce(identity, min, typemin, A, (1,3,2))

@timev VectorizedStatistics._vsum(A, (1,2,3));
@timev VectorizedStatistics._vsum(A, (1,3,2));
@timev VectorizedStatistics._vsum(A, (2,1,3));
@timev VectorizedStatistics._vsum(A, (2,3,1));
@timev VectorizedStatistics._vsum(A, (3,1,2));
@timev VectorizedStatistics._vsum(A, (3,2,1));

@timev vvmapreduce3(identity, min, typemin, A, (1,2,3));
@timev vvmapreduce3(identity, min, typemin, A, (1,3,2));
@timev vvmapreduce3(identity, min, typemin, A, (2,1,3));
@timev vvmapreduce3(identity, min, typemin, A, (2,3,1));
@timev vvmapreduce3(identity, min, typemin, A, (3,1,2));
@timev vvmapreduce3(identity, min, typemin, A, (3,2,1));

@timev vvmapreduce(identity, min, typemin, A, (1,2));
@timev vvmapreduce(identity, min, typemin, A, (1,3));
@timev vvmapreduce(identity, min, typemin, A, (1,4));
@timev vvmapreduce(identity, min, typemin, A, (2,1));
@timev vvmapreduce(identity, min, typemin, A, (2,3));
@timev vvmapreduce(identity, min, typemin, A, (2,4));
@timev vvmapreduce(identity, min, typemin, A, (3,1));
@timev vvmapreduce(identity, min, typemin, A, (3,2));
@timev vvmapreduce(identity, min, typemin, A, (3,4));
@timev vvmapreduce(identity, min, typemin, A, (4,1));
@timev vvmapreduce(identity, min, typemin, A, (4,2));
@timev vvmapreduce(identity, min, typemin, A, (4,3));

@timev B = vvmapreduce(identity, min, typemax, A, dims)
@timev vvmapreduce!(identity, min, typemax, B, A)
@code_typed _vvmapreduce!(identity, +, zero, B, A, dims)

################
using Static
using LoopVectorization
include("/nfs/site/home/aradclif/aradclif/.julia/dev/VectorizedReduction/src/lvreduce_v3.jl");
include("/nfs/site/home/aradclif/aradclif/.julia/dev/VectorizedReduction/src/vlogsumexp.jl");
using BenchmarkTools
#### Demonstrative experiments of branching and work functions
# It seems that the at reaching the terminal branching function, each
# of the work function calls which could result are taken to completion (i.e. they are compiled).
# Hence, assuming the same `f`, `op`, `init`, and array dimension N,
# (1,2,3) results in the path
# (StaticInt{1},Int,Int)
# (StaticInt{1},StaticInt{2},Int)
# (StaticInt{1},StaticInt{2},StaticInt{3}), (StaticInt{1},StaticInt{2},StaticInt{4})
# being fully compiled, such that (1,2,4) involves no compilation.
# That the work functions for (1,2,3), (1,2,4), (1,3,4) are all compiled on
# the first call, and that the compiler recognizes that permutations of these combinations
# would result in identical functions, seems to be supported by the fact that
# the first call for (1,2,3) has a substantially larger time and memory overhead.
# Compilation of both ends of the (1,2,3) path is certain. It is readily demonstrated by
# the (1,2,3), then (1,2,4) call sequence. On the other hand, the recognition of
# permutations of (1,2,3) such as (2,1,3), (1,3,2), etc. by the compiler as identical
# work functions is strongly supported. The compilation costs incurred
# on these calls should come from the need to compile the branching functions on those
# paths, and, presumably, also some costs associated with verifying that the proposed work
# function would in fact be identical to and already-compiled work function.
# Now it can be concluded that there will only be N choose M work functions,
# but there will still be M-permutations of N branching functions.
# Mathematically:
# N! / (M! * (N - M)!) work functions at most, i.e. all the combinations of M
# reduction indices from N total indices. For a given N, M, these are all compiled on the first call.
# N! / (N - M)! branching functions, at most, i.e. all permutations of each
# M-combination which could be formed from a set of N indices. These branches are incrementally
# compiled, but it exposes a fundamental issue: combinatorial explosion in the number of
# branches, which in fact can be avoided by sorting the dimensions upon entry to
# the mapreduce call. This ensures that permutations do not matter (by sorting dims),
# such that the number of branches functions is limited to N choose M as well.
# The downside is that there is a cost associated with sorting the dims tuple.
# As it incurs overhead of approximately 90ns, which, for length(A) = 625,
# is substantial (increase in time by factor of 1.33, and 80 bytes).
# Perhaps one should just make it the caller's responsibility to avoid
# this unnecessary compilation. It is difficult to imagine precisely what would motivate
# a scenario in which one might supply all the permutations of a given dims tuple.
####
# My preference is for making the caller responsible, but making them aware
# that randomly permuting a given combination of dims incurs a real cost.
# In other words, code such as below is to be avoided, unless there is a good reason.
# As the dimensionality increases, it silly things such as below could be very
# damaging to performance. Again, I emphasize that I cannot forsee a logical
# reason why a user would cycle through all the permutations.
using Random
# dims = (1,2,3) # The only combination
for _ = 1:10
    pdims = Tuple(randcycle(3))
    @timev vvmapreduce(identity, min, typemax, A, pdims)
end
binomial(4, 3)
permutations(N, M) = binomial(N, M) * factorial(M)
A′ = rand(3,3,3,3,3,3);
# dims = (1,2,3,4,5) # The only combination
for _ = 1:10
    pdims = Tuple(randcycle(5))
    @timev vvmapreduce(identity, min, typemax, A′, pdims)
end
####
A = rand(5,5,5,5);
# Test case:
# (1,2,3)    : 1 work, 3 branch
# (2,1,3)    : 2 branch
# (2,3,1)    : 1 branch
# Results
# 0.975314 seconds (5.02 M allocations: 246.505 MiB, 5.83% gc time, 100.00% compilation time)
# 0.407821 seconds (1.75 M allocations: 81.941 MiB, 8.08% gc time, 100.00% compilation time)
# 0.222833 seconds (744.10 k allocations: 33.510 MiB, 100.00% compilation time)
@timev vvmapreduce(identity, min, typemax, A, (1,2,3));
@timev vvmapreduce(identity, min, typemax, A, (2,1,3));
@timev vvmapreduce(identity, min, typemax, A, (2,3,1));
# (1,2,4)    : 1 work, 1 branch
@timev vvmapreduce(identity, min, typemax, A, (1,2,4));
# (2,1,4)    : 1 branch
@timev vvmapreduce(identity, min, typemax, A, (2,1,4));
# (2,3,4)    : 1 branch
@timev vvmapreduce(identity, min, typemax, A, (2,3,4));
# (1,4,2)    : 1 work, 1 branch
@timev vvmapreduce(identity, min, typemax, A, (1,4,2));
# (1,4,3)    : 1 branch
@timev vvmapreduce(identity, min, typemax, A, (1,4,3));
# (1,3,4)    : 1 work, 1 branch
@timev vvmapreduce(identity, min, typemax, A, (1,3,4));
# Results, fresh REPL
# 2.060194 seconds (8.22 M allocations: 415.810 MiB, 4.63% gc time, 100.00% compilation time)
# 0.751011 seconds (3.00 M allocations: 150.825 MiB, 3.43% gc time, 100.00% compilation time)
# 0.548857 seconds (1.71 M allocations: 86.841 MiB, 3.58% gc time, 100.00% compilation time)
# 0.000008 seconds (4 allocations: 208 bytes)
# 0.000008 seconds (4 allocations: 208 bytes)
# 0.000008 seconds (4 allocations: 208 bytes)

# Test case 2:
# (1,2,3)    : 1 work, 3 branch
# (1,2,4)    : 1 work, 1 branch
# (2,1,4)    : 2 branch
# (2,1,3)    : 1 branch
@timev vvmapreduce(identity, max, typemax, A, (1,2,3));
@timev vvmapreduce(identity, max, typemax, A, (1,2,4));
@timev vvmapreduce(identity, max, typemax, A, (2,1,4));
@timev vvmapreduce(identity, max, typemax, A, (2,1,3));
# Results
# 0.771038 seconds (4.53 M allocations: 219.878 MiB, 4.69% gc time, 100.00% compilation time)
# 0.000007 seconds (4 allocations: 208 bytes)
# 0.387111 seconds (1.76 M allocations: 82.305 MiB, 4.38% gc time, 100.00% compilation time)
# 0.000007 seconds (4 allocations: 208 bytes)

# Test case:
# does the non-generated function specialize on the types in the tuple? yes.
# does the generated function specialize on the types in the tuple? yes, at least in this case.
function bf(dims::D) where {M, D<:Tuple{Vararg{Integer, M}}}
    M + 1
end
d1 = (1,2,3)
d2 = (StaticInt{1}(),2,3)
d3 = (StaticInt{1}(),StaticInt{2}(),3)
d4 = (StaticInt{1}(),StaticInt{2}(),StaticInt{3}())
d6 = (StaticInt{1}(),StaticInt{3}(),2)
d5 = (StaticInt{1}(),StaticInt{3}(),StaticInt{2}())
@timev bf(d1)
@timev bf(d2)
@timev bf(d3)
@timev bf(d4)
@timev bf(d5)
@timev bf(d6)

@timev vvmapreduce(identity, +, zero, A, (1,2,3));
@timev vvmapreduce(identity, +, zero, A, (2,1,3));
@timev vvmapreduce(identity, +, zero, A, (2,3,1));
# (1,2,4)    : 1 work, 1 branch
@timev vvmapreduce(identity, +, zero, A, (1,2,4));
# (2,1,4)    : 1 branch
@timev vvmapreduce(identity, +, zero, A, (2,1,4));
# (2,3,4)    : 1 work, 1 branch
@timev vvmapreduce(identity, +, zero, A, (2,3,4));
# (1,4,2)    : 1 work, 1 branch
@timev vvmapreduce(identity, +, zero, A, (1,4,2));
# (1,4,3)    : 1 work, 1 branch
@timev vvmapreduce(identity, +, zero, A, (1,4,3));
# (1,3,4)    : 1 work, 1 branch
@timev vvmapreduce(identity, +, zero, A, (1,3,4));


################
function namedf2(f::F) where {F<:Function}
    if F <: Base.Fix2
        sym = gensym()
        φ = f.f
        x = f.x
        return @eval function $sym(y) $φ(y, $x) end
    else
        return f
    end
end

function namedf3(f::F) where {F<:Function}
    F <: Base.Fix2 ? begin @eval function $(gensym())(y) $(f.f)(y, $(f.x)) end end : f
end

################################################################
#### 2022-03-30: p. 62 extra 1-3
# Given N, M, at the terminal branching function, how many work functions will be compiled
# if the dims combination is new?
# In this context, new is defined as a combination that has not yet been compiled
# as the result of (1) a direct call (i.e. dims is just a permutation of some previous dims)
# or (2) an indirect call (i.e. some previous dims led to branching which inadvertently caused
# compilation, but said dims targeted a different combination).
# To illustrate with examples will clarify, using N=4, M=3
# An example of (1): consider a call made using dims=(1,2,3), then, consider a second call
# made using dims=(2,1,3). The second call is just a permutation of the dims from the first
# call, thus, branching aside, the work function is the same.
# An example of (2): consider a call made using dims=(1,2,3), then, consider a second call
# made using dims=(1,2,4). The second call corresponds to a different work function,
# but said work function would have been (inadvertently) compiled on the first call.
####
# At the mᵗʰ branching function call, there are (N - m) choose (M - m) remaining combinations.
# At the terminal branching function, all but one dimension is determined, thus
# there are (N - (M - 1)) choose 1 remaining ways to choose. Hence, for a new
# combination, (N - M + 1)! / 1!(N - M)! work functions will be compiled.
# The number of work functions compiled for a single combination as a fraction of the
# total work functions necessitated for all combinations is therefore M!(N - M + 1)! / N!.
# This fractional coverage experiences a minimum at M = N ÷ 2 + 1, but that is not
# necessarily relevant.
# What one should take from this is that given N, the number of work functions
# compiled by a single call decreases monotonically with increasing M,
# following (N - M + 1)! / 1!(N - M)!
# If one assumed (don't!) equal compilation time irrespective of M, then given N, the
# longest compilation for a single call would be at the smallest M. But, equal
# compilation time is an obviously false premise, hence, some additional modeling
# would be needed to determine the value of M which maximizes compilation time.

remaining(N, M, m) = binomial(N - m, M - m)
lastremaining(N, M) = remaining(N, M, M - 1)
# lastremaining2(N, M) = factorial(N - M + 1) ÷ factorial(N - M)
frac(N, M) = lastremaining(N, M) // binomial(N, M)
N = 7
map(M -> (M, binomial(N, M), lastremaining(N, M), frac(N, M)), 1:N)
a = map(N -> map(M -> (M, binomial(N, M), lastremaining(N, M), frac(N, M)), 1:N), 5:7)
