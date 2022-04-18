# VectorizedReduction

## Installation

```julia
using Pkg
Pkg.add("VectorizedReduction")
```

## Usage

This library provides "vectorized" (with/without multithreading) versions of the following functions
1. `mapreduce` and common derived functions: `reduce`, `sum`, `prod`, `minimum`, `maximum`, `extrema`
2. `count`, `any`, `all`
3. `findmin`, `findmin`, `argmin`, `argmax`
4. `logsumexp`, `softmax`, `logsoftmax` ("safe" versions: avoid underflow/overflow)

The naming convention is as follows: a vectorized (without threading) version is prefixed by `v`, and a vectorized with threading version is prefixed by `vt`.
There is a single exception to this rule: vectorized (without threading) versions of the functions listen in 1. are prefixed by `vv` in order to avoid name collisions with LoopVectorization and VectorizedStatistics.

## Motivation

When writing numerical code, one may with to perform a reduction, perhaps across multiple dimensions, as the most natural means of expressing the relevant mathematical operation.
For readers well-acquainted with LoopVectorization.jl, the thought immediately comes to mind: writing out the loops will inevitably be a large performance gain. Thus, in that neverending pursuit of fast code, we write the loops -- but this produces specific code, tailored to the dimensions of the problem.
Instead, we might have liked to write generic code, parameterizing our function with an index set of dimensions. This package attempts to resolve this perpetual dilemma using metaprogramming. The hope is that the next time one asks the question: is it worthwhile to write the loops (gain performance, lose genericity), or can I make do with the Base implementation? that one can confidently reach for one of the "vectorized" versions provided by this package.

Now, before diving into other interesting topics, a few salient comments. Foremost, this package is not intended as a univeral replacement for Base implementations of the same functions. The scope of functions provided by this package is limited to numerical code, and subject to all restrictions imposed by LoopVectorization (the user is encouraged to familiarize themselves with said package).
Moreover, there exist limitations of `vfindmin` and friends which are not yet resolved (a workaround is provided, but not without a performance cost). That being said, the user may find useful the expanded capabilities of `vfindmin`/`vfindmax` which are not yet part of Base (an aspect the author hopes to rectify). The user is encouraged to read the additional commentary below, but it is not strictly necessary.

## Commentary
### Why provide `dims` directly, rather than as kwargs? (an optional performance enhancement for small arrays)
To define a multi-dimensional mapreduce, one specifies a index set of dimensions, `dims` over which the reduction will occur. In Base, one passes `dims` using keyword arguments, e.g. `sum(A, dims=(2,4))`. However, this sometimes incurs an overhead of ~20ns. Admittedly, small, and in almost all cases negligible, but, if one wishes to ensure that such costs are avoided, the interface also supports direct calls which provide `init` and `dims` as positional arguments.

### When to consider replacing any/all with vectorized versions
Assumptions: x is inherently unordered, such that the probablilty of a "success" is independent of the index (i.e. independent of a random permutation of the effective vector over which the reduction occurs).
This is a very reasonable assumption for certain types of data, e.g. Monte Carlo simulations. However, for cases in which there is some inherent order (e.g. a solution to an ODE, or even very simply, `cos.(-2π:0.1:2π)`), then the analysis below does not hold. If inherent order exists, then the probability of success depends on said ordering -- the caller must then exercise judgment based on where the first success might land (if at all); as this is inevitably problem-specific, I am unable to offer general advice.

For inherently unordered data:
Define `p` as the probability of "success", i.e. the probability of `true` w.r.t. `any` and the probability of `false` w.r.t. `all`.
The cumulative probability of evaluating all elements, `Pr(x ≤ 0)` is
```julia
binomial(n, 0) * p^0 * (1 - p)^(n - 0) # (1 - p)^n
```
Define a linearized cost model, with `t` the time required to evaluate one element, and `n` the length of the vector. Denote `t₀` as the non-vectorized evaluation time per element, and `tᵥ` as the vectorized evaluation time per element. A crude estimate for the expected cost of the call is therefore
```julia
t₀ * n * (1 - p)^n
```
Thus, the point at which non-vectorized evaluation is optimal is
```julia
t₀ * (1 - p)^n < tᵥ
```
Or, rearranging: non-vectorized is optimal when `p > 1 - (tᵥ/t₀)^(1/n)`. Intuitively, as `(tᵥ/t₀)` becomes smaller, larger `p` is needed to make the non-vectorized option optimal.
Holding `(tᵥ/t₀)` constant, increasing `n` results in a rapid decrease in the `p` required for the non-vectorized option to be optimal. Consider the following examples, denoting `r = (tᵥ/t₀)`
```julia
julia> p(r, n) = 1 - r^(1/n)
p (generic function with 1 method)

julia> p.(.1, 10 .^ (1:4))
4-element Vector{Float64}:
 0.2056717652757185
 0.02276277904418933
 0.0022999361774467264
 0.0002302320018434667

julia> p.(.01, 10 .^ (1:4))
4-element Vector{Float64}:
 0.36904265551980675
 0.045007413978564004
 0.004594582648473011
 0.0004604109969121861
```
However, due to the current implementation details of Base `any`/`all`, early breakout occurs only when the reduction is being carried out across the entire array (i.e. does not occur when reducing over a subset of dimensions). Thus, the current advice is to use `vany`/`vall` unless one is reducing over the entire array, and even then, one should consider the `p` and `n` for one's problem.

## Examples
### Simple examples

<details>
 <summaryClick me! ></summary>
<p>

A very simple comparison.
```julia
julia> A = rand(5,5,5,5);

julia> @benchmark mapreduce($abs2, $+, $A, dims=$(1,2,4))
BenchmarkTools.Trial: 10000 samples with 133 evaluations.
 Range (min … max):  661.038 ns … 139.234 μs  ┊ GC (min … max): 0.00% … 99.24%
 Time  (median):     746.880 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   798.069 ns ±   1.957 μs  ┊ GC (mean ± σ):  3.46% ±  1.40%

 Memory estimate: 368 bytes, allocs estimate: 8.

julia> @benchmark vvmapreduce($abs2, $+, $A, dims=$(1,2,4))
BenchmarkTools.Trial: 10000 samples with 788 evaluations.
 Range (min … max):  160.538 ns …  29.430 μs  ┊ GC (min … max):  0.00% … 99.11%
 Time  (median):     203.479 ns               ┊ GC (median):     0.00%
 Time  (mean ± σ):   212.916 ns ± 761.848 ns  ┊ GC (mean ± σ):  10.68% ±  2.97%

 Memory estimate: 240 bytes, allocs estimate: 6.

julia> @benchmark extrema($A, dims=$(1,2))
BenchmarkTools.Trial: 10000 samples with 9 evaluations.
 Range (min … max):  2.813 μs …   5.827 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     2.990 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.039 μs ± 149.676 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

 Memory estimate: 960 bytes, allocs estimate: 14.

julia> @benchmark vvextrema($A, dims=$(1,2))
BenchmarkTools.Trial: 10000 samples with 202 evaluations.
 Range (min … max):  381.743 ns … 86.288 μs  ┊ GC (min … max):  0.00% … 99.05%
 Time  (median):     689.658 ns              ┊ GC (median):     0.00%
 Time  (mean ± σ):   712.113 ns ±  2.851 μs  ┊ GC (mean ± σ):  13.84% ±  3.43%

 Memory estimate: 1.19 KiB, allocs estimate: 8.
```
</p>
</details>

### Varargs examples
<details>
 <summaryClick me! ></summary>
<p>

These are somewhat standard fare, but can be quite convenient for expressing
certain Bayesian computations.
```julia
julia> A1, A2, A3, A4 = rand(5,5,5,5), rand(5,5,5,5), rand(5,5,5,5), rand(5,5,5,5);

julia> @benchmark mapreduce($+, $+, $A1, $A2, $A3, $A4, dims=$(1,2,4))
BenchmarkTools.Trial: 10000 samples with 10 evaluations.
 Range (min … max):  1.597 μs …  1.181 ms  ┊ GC (min … max): 0.00% … 97.71%
 Time  (median):     1.867 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.257 μs ± 14.216 μs  ┊ GC (mean ± σ):  8.56% ±  1.38%

 Memory estimate: 5.66 KiB, allocs estimate: 14.

julia> @benchmark vvmapreduce($+, $+, $A1, $A2, $A3, $A4, dims=$(1,2,4))
BenchmarkTools.Trial: 10000 samples with 203 evaluations.
 Range (min … max):  384.768 ns … 150.041 μs  ┊ GC (min … max): 0.00% … 99.57%
 Time  (median):     437.601 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   478.179 ns ±   2.117 μs  ┊ GC (mean ± σ):  7.50% ±  1.72%

 Memory estimate: 304 bytes, allocs estimate: 6.

# And for really strange stuff (e.g. posterior predictive transformations)
julia> @benchmark vvmapreduce((x,y,z) -> ifelse(x*y+z ≥ 1, 1, 0), +, $A1, $A2, $A3)
BenchmarkTools.Trial: 10000 samples with 198 evaluations.
 Range (min … max):  438.126 ns …  5.704 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     439.995 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   442.020 ns ± 63.038 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

 Memory estimate: 0 bytes, allocs estimate: 0.

# using ifelse for just a boolean is quite slow, but the above is just for demonstration
julia> @benchmark vvmapreduce((x,y,z) -> ≥(x*y+z, 1), +, $A1, $A2, $A3)
BenchmarkTools.Trial: 10000 samples with 975 evaluations.
 Range (min … max):  70.558 ns …  2.085 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     70.888 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   71.425 ns ± 23.489 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

 Memory estimate: 0 bytes, allocs estimate: 0.

# What I mean by posterior predictive transformation? Well, one might encounter
# this in Bayesian model checking, which provides a convenient example.
# If one wishes to compute the Pr = ∫∫𝕀(T(yʳᵉᵖ, θ) ≥ T(y, θ))p(yʳᵉᵖ|θ)p(θ|y)dyʳᵉᵖdθ
# Let's imagine that A1 represents T(yʳᵉᵖ, θ) and A2 represents T(y, θ)
# i.e. the test variable samples computed as a functional of the Markov chain (samples of θ)
# Then, Pr is computed as
vvmapreduce(≥, +, A1, A2) / length(A1)
# Or, if only the probability is of interest, and we do not wish to use the functionals
# for any other purpose, we could compute it as:
vvmapreduce((x, y) -> ≥(f(x), f(y)), +, A1, A2) / length(A1)
# where `f` is the functional of interest, e.g.
vvmapreduce((x, y) -> ≥(abs2(x), abs2(y)), +, A1, A2) / length(A1)

# One can also express commonly encountered reductions with ease;
# these will be fused once a post-reduction operator can be specified
# Mean squared error
vvmapreduce((x, y) -> abs2(x - y), +, A1, A2, dims=(2,4)) ./ (size(A1, 2) * size(A1, 4))
# Euclidean distance
(√).(vvmapreduce((x, y) -> abs2(x - y), +, A1, A2, dims=(2,4)))
```
</p>
</details>

### `findmin`/`findmax` examples
<details>
 <summaryClick me! ></summary>
<p>
# Examples of extended syntax (applies to `findmin`, `findmax`, `argmin, `argmax`): `findmin(f, A; dims)`, `findmin(f, A...; dims)`. In the former case, `f` : ℝ → ℝ; in the latter, `f` : ℝᴺ → ℝ.
```julia
# Easy to express without the extended syntax, but not efficient.
julia> B1, B2, B3 = rand(5,5,5,5), rand(5,5,5,5), rand(5,5,5,5);

julia> B′ = @. B1 + B2 + B3;

julia> findmax(B′) == vfindmax(+, B1, B2, B3)
true

julia> @benchmark findmin(@. $B1 + $B2 + $B3)
@benchmark vfindmin(+, $B1, $B2, $B3)
BenchmarkTools.Trial: 10000 samples with 8 evaluations.
 Range (min … max):  3.905 μs … 943.922 μs  ┊ GC (min … max): 0.00% … 94.06%
 Time  (median):     4.011 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.128 μs ±   9.418 μs  ┊ GC (mean ± σ):  2.15% ±  0.94%

 Memory estimate: 5.11 KiB, allocs estimate: 2.

julia> @benchmark vfindmin(+, $B1, $B2, $B3)
BenchmarkTools.Trial: 10000 samples with 943 evaluations.
 Range (min … max):  100.346 ns … 151.376 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     100.821 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   100.866 ns ±   0.651 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

 Memory estimate: 0 bytes, allocs estimate: 0.

# A multidimensional example

julia> @benchmark findmin((@. abs2($B1) * $B2 + $B3), dims=$(3,4))
BenchmarkTools.Trial: 10000 samples with 7 evaluations.
 Range (min … max):  4.026 μs …  1.132 ms  ┊ GC (min … max): 0.00% … 94.30%
 Time  (median):     4.311 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.494 μs ± 11.335 μs  ┊ GC (mean ± σ):  2.37% ±  0.94%

 Memory estimate: 6.55 KiB, allocs estimate: 12.

julia> @benchmark vfindmin((x, y, z) -> abs2(x) * y + z, $B1, $B2, $B3, dims=$(3,4))
BenchmarkTools.Trial: 10000 samples with 168 evaluations.
 Range (min … max):  623.071 ns … 118.436 μs  ┊ GC (min … max):  0.00% … 99.10%
 Time  (median):       1.053 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):     1.101 μs ±   4.003 μs  ┊ GC (mean ± σ):  12.56% ±  3.43%

 Memory estimate: 1.62 KiB, allocs estimate: 9.
```
</p>
</details>


## Acknowledgments
The original motivation for this work was a vectorized & multithreaded multi-dimensional findmin, taking a variable number of array arguments -- it's a long story, but the similarity between findmin and mapreduce motivated a broad approach. My initial attempt (visible in the /attic) did not deliver all the performance possible -- this was only apparent through comparison to C. Elrod's approach to multidimensional forms in VectorizedStatistics. Having fully appreciated the beauty of branching through @generated functions, I decided to take a tour of some low-hanging fruit -- this package is the result.

## Future work
1. post-reduction operators
2. reductions over index subsets within a dimension.

## Elsewhere
* [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) back-end for this package.
* [Tullio.jl](https://github.com/mcabbott/Tullio.jl) express any of the reductions using index notation