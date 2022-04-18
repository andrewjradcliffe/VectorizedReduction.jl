# VectorizedReduction

## Installation

```julia
using Pkg
Pkg.add("VectorizedReduction")
```

## Usage

This library provides "vectorized" (with/without multithreading) versions of the following functions
1. `mapreduce` and common derived functions: `reduce`, `sum`, `prod`, `minimum`, `maximum`, `extrema`, `count`
2. `any`, `all` (listed separately to emphasize theoretical considerations on applicability)
3. `findmin`, `findmin`, `argmin`, `argmax`
4. `logsumexp`, `softmax`, `logsoftmax`

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
This is a very reasonable assumption for certain types of data, e.g. Monte Carlo simulations. However, for cases in which there is some inherent order (e.g. a solution to an ODE, or even very simply, `map(exp, 0.0:0.1:10.0)`), then the analysis below does not hold. If inherent order exists, then the probability of success depends on said ordering -- the caller must then exercise judgment based on where the first success might land (if at all); as this is inevitably problem-specific, I am unable to offer general advice.

For inherently unordered data:
Define `p` as the probability of success, i.e. the probability of `true` w.r.t. `any` and the probability of `false` w.r.t. `all`.
The cumulative probability of evaluating all elements:
```math
Pr(x \leq 0) = {n \\choose 0} p^0 (1 - p)^{n-0} = (1 - p)^n
```
Define a linearized cost model, with `t` the time required to evaluate one element, and `n` the length of the vector. Denote `t_0` as the non-vectorized evaluation time per element, and `t_v` as the vectorized evaluation time per element. A crude estimate for the expected cost of the call is therefore
```math
t_{0} n (1 - p)^n
```
<img src="https://render.githubusercontent.com/render/math?math={t_{0} n (1 - p)^{n}}#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\color{white}t_{0} n (1 - p)^{n}}#gh-dark-mode-only">