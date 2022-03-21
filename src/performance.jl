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
