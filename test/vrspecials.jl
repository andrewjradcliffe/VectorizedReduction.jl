# Tests of some unique syntax, and some shared syntax

@testset "vvmapreduce" begin
    A = rand(5,5,5,5)
    @testset "test reductions over region: $region" for region in Any[
        1, 2, 3, 4, 5, (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
        (1, 2, 3), (1, 3, 4), (2, 3, 4), (1, 2, 3, 4), :]
        @test vvmapreduce(abs2, +, A, dims=region) ≈ mapreduce(abs2, +, A, dims=region)
        @test vvmapreduce(cos, *, A, dims=region) ≈ mapreduce(cos, *, A, dims=region)
        @test vvprod(log, A, dims=region) ≈ prod(log, A, dims=region)
        @test vvminimum(sin, A, dims=region) ≈ minimum(sin, A, dims=region)
        @test vvmaximum(sin, A, dims=region) ≈ maximum(sin, A, dims=region)
        if region !== Colon()
            @test all((((x1, y1), (x2, y2)),) -> x1 ≈ x2 && y1 ≈ y2, zip(vvextrema(sin, A, dims=region), extrema(sin, A, dims=region)))
        else
            @test all(vvextrema(sin, A, dims=region) .≈ extrema(sin, A, dims=region))
        end
    end
end
@testset "vvmapreduce_vararg" begin
    A1 = rand(5,5,5,5)
    A2 = rand(5,5,5,5)
    A3 = rand(5,5,5,5)
    A4 = rand(1:10, 5,5,5,5)
    as = (A1, A2, A3)
    f3(x, y, z) = x * y + z
    f4(w, x, y, z) = inv(w) * exp(abs2(x - y) * z)
    @testset "test reductions over region: $region" for region in Any[
        1, 2, 3, 4, 5, (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
        (1, 2, 3), (1, 3, 4), (2, 3, 4), (1, 2, 3, 4), (1,2,3,4,5), :]
        @test vvmapreduce(+, +, as, dims=region) ≈ mapreduce(+, +, A1, A2, A3, dims=region)
        @test vvmapreduce(*, +, as, dims=region) ≈ mapreduce(*, +, A1, A2, A3, dims=region)
        @test vvmapreduce(+, *, as, dims=region) ≈ mapreduce(+, *, A1, A2, A3, dims=region)
        @test vvmapreduce(f3, *, as, dims=region) ≈ mapreduce(f3, *, A1, A2, A3, dims=region)

        # slurping
        @test vvmapreduce(+, +, A1, A2, A3, dims=region) ≈ mapreduce(+, +, A1, A2, A3, dims=region)
        @test vvmapreduce(*, +, A1, A2, A3, dims=region) ≈ mapreduce(*, +, A1, A2, A3, dims=region)
        @test vvmapreduce(+, *, A1, A2, A3, dims=region) ≈ mapreduce(+, *, A1, A2, A3, dims=region)
        @test vvmapreduce(f3, *, A1, A2, A3, dims=region) ≈ mapreduce(f3, *, A1, A2, A3, dims=region)
        @test vvmapreduce(f4, *, A1, A2, A3, A3, dims=region) ≈ mapreduce(f4, *, A1, A2, A3, A3, dims=region)
        # Tests of variably typed arrays
        @test vvmapreduce(+, +, A1, A2, dims=region) ≈ mapreduce(+, +, A1, A2, dims=region)
        @test vvmapreduce(+, +, A1, A4, dims=region) ≈ mapreduce(+, +, A1, A4, dims=region)
        @test vvmapreduce(*, +, A1, A2, dims=region) ≈ mapreduce(*, +, A1, A2, dims=region)
        @test vvmapreduce(+, *, A1, A4, dims=region) ≈ mapreduce(+, *, A1, A4, dims=region)
    end
    # interface tests
    r = mapreduce(*, +, A1, A2, A3)
    @test r ≈ vvmapreduce(*, +, zero, A1, A2, A3)
    @test r ≈ vvmapreduce(*, +, A1, A2, A3)
    @test r ≈ vvmapreduce(*, +, A1, A2, A3, dims=:)
    @test r ≈ vvmapreduce(*, +, A1, A2, A3, dims=:, init=0)
    @test r ≈ vvmapreduce(*, +, A1, A2, A3, dims=:, init=zero)
    @test r ≈ vvmapreduce(*, +, as)
    # # And for really strange stuff (e.g. posterior predictive transformations)
    # @benchmark vvmapreduce((x,y,z) -> ifelse(x*y+z ≥ 1, 1, 0), +, A1, A2, A3)
    # @benchmark vvmapreduce((x,y,z) -> ifelse(x*y+z ≥ 1, 1, 0), +, A1, A2, A3, dims=(2,3,4))
    # # using ifelse for just a boolean is quite slow, but the above is just for demonstration
    # @benchmark vvmapreduce(≥, +, A1, A2)
    # @benchmark vvmapreduce((x,y,z) -> ≥(x*y+z, 1), +, A1, A2, A3)
    # @benchmark vvmapreduce((x,y,z) -> ≥(x*y+z, 1), +, A1, A2, A3, dims=(2,3,4))
    # @benchmark mapreduce((x,y,z) -> ≥(x*y+z, 1), +, A1, A2, A3)
    # # What I mean by posterior predictive transformation? Well, one might encounter
    # # this in Bayesian model checking, which provides a convenient example.
    # # If one wishes to compute the Pr = ∫∫𝕀(T(yʳᵉᵖ, θ) ≥ T(y, θ))p(yʳᵉᵖ|θ)p(θ|y)dyʳᵉᵖdθ
    # # Let's imagine that A1 represents T(yʳᵉᵖ, θ) and A2 represents T(y, θ)
    # # i.e. the test variable samples computed as a functional of the Markov chain (samples of θ)
    # Then, Pr is computed as
    # vvmapreduce(≥, +, A1, A2) / length(A1)
    @test mapreduce(≥, +, A1, A2) ≈ vvmapreduce(≥, +, A1, A2)
    # Or, if only the probability is of interest, and we do not wish to use the functionals
    # for any other purpose, we could compute it as:
    # vvmapreduce((x, y) -> ≥(f(x), f(y)), +, A1, A2)
    # where `f` is the functional of interest, e.g.
    f(x, y) = ≥(abs2(x), abs2(y))
    # r = mapreduce((x, y) -> ≥(abs2(x), abs2(y)), +, A1, A2)
    r = mapreduce(f, +, A1, A2)
    # @test r ≈ vvmapreduce((x, y) -> ≥(abs2(x), abs2(y)), +, A1, A2)
    @test r ≈ vvmapreduce(f, +, A1, A2)
    # R = mapreduce((x, y) -> ≥(abs2(x), abs2(y)), +, A1, A2, dims=(2,3,4))
    # @test R ≈ vvmapreduce((x, y) -> ≥(abs2(x), abs2(y)), +, A1, A2, dims=(2,3,4))
    R = mapreduce(f, +, A1, A2, dims=(2,3,4))
    @test R ≈ vvmapreduce(f, +, A1, A2, dims=(2,3,4))
    # One can also express commonly encountered reductions with ease;
    # these will be fused once a post-reduction operator can be specified
    # MSE
    sqdiff(x, y) = abs2(x -y)
    # B = vvmapreduce((x, y) -> abs2(x - y), +, A1, A2, dims=(2,4)) ./ (size(A1, 2) * size(A1, 4) )
    # @test B ≈ mapreduce((x, y) -> abs2(x - y), +, A1, A2, dims=(2,4)) ./ (size(A1, 2) * size(A1, 4) )
    B = vvmapreduce(sqdiff, +, A1, A2, dims=(2,4)) ./ (size(A1, 2) * size(A1, 4))
    @test B ≈ mapreduce(sqdiff, +, A1, A2, dims=(2,4)) ./ (size(A1, 2) * size(A1, 4))
    # Euclidean distance
    # B = (√).(vvmapreduce((x, y) -> abs2(x - y), +, A1, A2, dims=(2,4)))
    # @test B ≈ (√).(mapreduce((x, y) -> abs2(x - y), +, A1, A2, dims=(2,4)))
    B = (√).(vvmapreduce(sqdiff, +, A1, A2, dims=(2,4)))
    @test B ≈ (√).(mapreduce(sqdiff, +, A1, A2, dims=(2,4)))
end
@testset "vfindminmax" begin
    # Simple
    A1 = rand(5,5)
    A2 = rand(5,5)
    A3 = rand(5,5)
    A′ = @. A1 + A2 + A3
    @test findmin(A′) == vfindmin(+, A1, A2, A3)
    @test findmin(A′, dims=2) == vfindmin(+, A1, A2, A3, dims=2)
    #
    v1 = rand(5)
    v2 = rand(5)
    v3 = rand(5)
    v′ = @. v1 + v2 + v3;
    @test findmin(v′) == vfindmin(+, v1, v2, v3)
    #
    A = rand(5,5,5,5)
    A′ = cos.(A)
    val1, ind1 = findmax(A′, dims=(2,4))
    val2, ind2 = vfindmax(cos, A, dims=(2,4))
    @test ind1 == ind2 && val1 ≈ val2
    #
    g(x) = ifelse(abs(x) ≥ 2, 100, 1)
    A = randn(5,5,5,5)
    A′ = g.(A)
    val1, ind1 = findmax(A′, dims=(2,4))
    val2, ind2 = vfindmax(g, A, dims=(2,4))
    @test ind1 == ind2 && val1 ≈ val2
end
@testset "vfindminmax_vararg" begin
    Random.seed!(1234)
    A1 = rand(5,5)
    A2 = rand(5,5)
    A3 = rand(5,5)
    A′ = @. A1 * A2 + A3;
    @test findmin(A′) == vfindmin((x, y, z) -> x * y + z, A1, A2, A3)
    val1, ind1 = findmin(A′, dims=2)
    val2, ind2 = vfindmin((x, y, z) -> x * y + z, A1, A2, A3, dims=2)
    @test ind1 == ind2 && val1 ≈ val2
    #
    B1, B2, B3 = randn(5,5,5,5), randn(5,5,5,5), randn(5,5,5,5);
    h(x, y, z) = ifelse(x ≥ .5, 100, 1) * y + √abs(z)
    B′ = h.(B1, B2, B3)
    val1, ind1 = findmax(B′, dims=(2,4))
    val2, ind2 = vfindmax(h, B1, B2, B3, dims=(2,4))
    @test ind1 == ind2 && val1 ≈ val2

    f3(x, y, z) = x * y + z
    @testset "test reductions over region: $region" for region in Any[
        2, 3, 4, 5, (1, 2), (2, 3), (2, 4), (3, 4),
        (2, 3), (3, 4), (2, 3, 4), (1, 2, 3, 4), (1,2,3,4,5), :]
        rval, rind = findmin(A′, dims=region)
        rval′, rind′ = vfindmin(f3, A1, A2, A3, dims=region)
        @test rval ≈ rval′
        @test rind == rind′
    end
end
