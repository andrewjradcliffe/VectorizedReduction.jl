#
# Date created: 2022-06-22
# Author: aradclif
#
#
############################################################################################

function vfindextrema(f::F, initmin::Iₘᵢₙ, initmax::Iₘₐₓ, A::AbstractArray{T, N}) where {F, Iₘᵢₙ, Iₘₐₓ, T, N}
    Tₒ = Base.promote_op(f, T)
    mn, mx = initmin(Tₒ), initmax(Tₒ)
    i_mn, i_mx = firstindex(A), firstindex(A)
    @turbo check_empty=true for i ∈ eachindex(A)
        v = f(A[i])
        newmin = v < mn
        newmax = v > mx
        i_mn = ifelse(newmin, i, i_mn)
        i_mx = ifelse(newmax, i, i_mx)
        mn = ifelse(newmin, v, mn)
        mx = ifelse(newmax, v, mx)
    end
    ((mn, i_mn), (mx, i_mx))
end
vfindextrema(f, A) = vfindextrema(f, typemax, typemin, A)
vfindextrema(A) = vfindextrema(identity, typemax, typemin, A)

_rf_findextrema((((fm₁, im₁), (fx₁, ix₁))), (((fm₂, im₂), (fx₂, ix₂)))) =
    ((Base.isgreater(fm₁, fm₂) ? (fm₂, im₂) : (fm₁, im₁)), (isless(fx₁, fx₂) ? (fx₂, ix₂) : (fx₁, ix₁)))
findextrema(f, domain) = mapfoldl(((k, v),) -> ((f(v), k), (f(v), k)), _rf_findextrema, pairs(domain))
findextrema(domain) = findextrema(identity, domain)

vfindextrema2(A) = vfindmin(A), vfindmax(A)
vfindextrema2(f, A) = vfindmin(f, A), vfindmax(f, A)

@benchmark vfindextrema($x)
@benchmark findextrema($x)
findextrema(x)
vfindextrema(x)
@benchmark vfindmin($x)
@benchmark vfindmax($x)
for i = 1:20
    for j = -1:1
        N = (1 << i) + j
        x = rand(N)
        println("N = $N; vfindextrema (single pass):")
        @btime vfindextrema($x)
        println("N = $N; vfindextrema2 (two passes):")
        @btime vfindextrema2($x)
    end
end
