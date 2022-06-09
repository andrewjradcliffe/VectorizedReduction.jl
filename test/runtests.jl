using VectorizedReduction
using Test

@testset "VectorizedReduction.jl" begin
    @testset "vvmapreduce" begin
        A = rand(5,5,5,5)
        @test vvmapreduce(abs2, +, A, dims=(1,3)) ≈ mapreduce(abs2, +, A, dims=(1,3))
        @test vvmapreduce(cos, *, A, dims=(2,4)) ≈ mapreduce(cos, *, A, dims=(2,4))
        @test vvprod(log, A, dims=1) ≈ prod(log, A, dims=1)
        @test vvminimum(sin, A, dims=(3,4)) ≈ minimum(sin, A, dims=(3,4))
        @test vvmaximum(sin, A, dims=(3,4)) ≈ maximum(sin, A, dims=(3,4))
    end
    @testset "vvmapreduce_vararg" begin
        A1 = rand(5,5,5,5)
        A2 = rand(5,5,5,5)
        A3 = rand(5,5,5,5)
        A4 = rand(1:10, 5,5,5,5)
        as = (A1, A2, A3)
        @test vvmapreduce(+, +, as, dims=(1,2,4)) ≈ mapreduce(+, +, A1, A2, A3, dims=(1,2,4))
        # Tests of variably typed arrays
        @test vvmapreduce(+, +, A1, A2, dims=(2,3,4)) ≈ mapreduce(+, +, A1, A2, dims=(2,3,4))
        @test vvmapreduce(+, +, A1, A4, dims=(2,3,4)) ≈ mapreduce(+, +, A1, A4, dims=(2,3,4))
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
        vvmapreduce(≥, +, A1, A2) / length(A1)
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
    end
    ############################################################################################
    @testset "from reducedim.jl tests" begin
        A = [1.0 5.0 6.0;
             5.0 2.0 4.0]
        for (tup, rval, rind) in [((1,), [1.0 2.0 4.0], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(2,3)]),
                                  ((2,), reshape([1.0,2.0], 2, 1), reshape([CartesianIndex(1,1),CartesianIndex(2,2)], 2, 1)),
                                  ((1,2), fill(1.0,1,1),fill(CartesianIndex(1,1),1,1))]
            @test vfindmin1(A, dims=tup) == (rval, rind)
            @test isequal(vvminimum(A, dims=tup), rval)
        end

        for (tup, rval, rind) in [((1,), [5.0 5.0 6.0], [CartesianIndex(2,1) CartesianIndex(1,2) CartesianIndex(1,3)]),
                                  ((2,), reshape([6.0,5.0], 2, 1), reshape([CartesianIndex(1,3),CartesianIndex(2,1)], 2, 1)),
                                  ((1,2), fill(6.0,1,1),fill(CartesianIndex(1,3),1,1))]
            @test vfindmax1(A, dims=tup) == (rval, rind)
            @test isequal(vvmaximum(A, dims=tup), rval)
        end
    end
    @testset "findmin/findmax transformed arguments, numeric values" begin
        A = [1.0 -5.0 -6.0;
             -5.0 2.0 4.0]
        TA = [((1,), [1.0 2.0 4.0], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(2,3)]),
              ((2,), reshape([1.0, 2.0], 2, 1), reshape([CartesianIndex(1,1), CartesianIndex(2,2)], 2, 1)),
              ((1,2), fill(1.0,1,1), fill(CartesianIndex(1,1),1,1))]
        TA2 = [((1,), [1.0 4.0 16.0], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(2,3)]),
               ((2,), reshape([1.0, 4.0], 2, 1), reshape([CartesianIndex(1,1), CartesianIndex(2,2)], 2, 1)),
               ((1,2), fill(1.0,1,1), fill(CartesianIndex(1,1),1,1))]
        TAc = [((1,), [0.28366218546322625 -0.4161468365471424 -0.6536436208636119], [CartesianIndex(2,1) CartesianIndex(2,2) CartesianIndex(2,3)]),
               ((2,), reshape([0.28366218546322625, -0.6536436208636119], 2, 1), reshape([CartesianIndex(1,2), CartesianIndex(2,3)], 2, 1)),
               ((1,2), fill(-0.6536436208636119,1,1), fill(CartesianIndex(2,3),1,1))]
        for (f, At) in ((abs, TA), (abs2, TA2), (cos, TAc))
            A′ = map(f, A)
            for (tup, rval, rind) in At
                (rval′, rind′) = vfindmin1(f, A, dims=tup)
                @test all(rval′ .≈ rval)
                @test rind′ == rind
                (rval′′, rind′′) = vfindmin1(A′, dims=tup)
                @test all(rval′ .≈ rval′′)
                @test rind′ == rind′′
            end
        end

        TA = [((1,), [5.0 5.0 6.0], [CartesianIndex(2,1) CartesianIndex(1,2) CartesianIndex(1,3)]),
              ((2,), reshape([6.0,5.0], 2, 1), reshape([CartesianIndex(1,3), CartesianIndex(2,1)], 2, 1)),
              ((1,2), fill(6.0,1,1),fill(CartesianIndex(1,3),1,1))]
        TA2 = [((1,), [25.0 25.0 36.0], [CartesianIndex(2,1) CartesianIndex(1,2) CartesianIndex(1,3)]),
               ((2,), reshape([36.0, 25.0], 2, 1), reshape([CartesianIndex(1,3), CartesianIndex(2,1)], 2, 1)),
               ((1,2), fill(36.0,1,1), fill(CartesianIndex(1,3),1,1))]
        TAc = [((1,), [0.5403023058681398 0.28366218546322625 0.960170286650366], [CartesianIndex(1,1) CartesianIndex(1,2) CartesianIndex(1,3)]),
               ((2,), reshape([0.960170286650366, 0.28366218546322625], 2, 1), reshape([CartesianIndex(1,3), CartesianIndex(2,1)], 2, 1)),
               ((1,2), fill(0.960170286650366,1,1), fill(CartesianIndex(1,3),1,1))]
        for (f, At) in ((abs, TA), (abs2, TA2), (cos, TAc))
            A′ = map(f, A)
            for (tup, rval, rind) in At
                (rval′, rind′) = vfindmax1(f, A, dims=tup)
                @test all(rval′ .≈ rval)
                @test rind′ == rind
                (rval′′, rind′′) = vfindmax1(A′, dims=tup)
                @test all(rval′ .≈ rval′′)
                @test rind′ == rind′′
            end
        end
    end

    @testset "NaN in findmin/findmax/minimum/maximum" begin
        A = [1.0 NaN 6.0;
             NaN 2.0 4.0]
        A′ = [-1.0 NaN -6.0;
              NaN -2.0 4.0]
        for (tup, rval, rind) in [((1,), [1.0 2.0 4.0], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(2,3)]),
                                  ((2,), reshape([1.0, 2.0], 2, 1), reshape([CartesianIndex(1,1),CartesianIndex(2,2)], 2, 1)),
                                  ((1,2), fill(1.0,1,1),fill(CartesianIndex(1,1),1,1))]
            @test isequal(vfindmin1(A, dims=tup), (rval, rind))
            @test isequal(vfindmin1(abs, A′, dims=tup), (rval, rind))
            @test isequal(vvminimum(A, dims=tup), rval)
            @test isequal(vvminimum(abs, A′, dims=tup), rval)
        end

        for (tup, rval, rind) in [((1,), [1.0 2.0 6.0], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(1,3)]),
                                  ((2,), reshape([6.0, 4.0], 2, 1), reshape([CartesianIndex(1,3),CartesianIndex(2,3)], 2, 1)),
                                  ((1,2), fill(6.0,1,1),fill(CartesianIndex(1,3),1,1))]
            @test isequal(vfindmax1(A, dims=tup), (rval, rind))
            @test isequal(vfindmax1(abs, A′, dims=tup), (rval, rind))
            @test isequal(vvmaximum(A, dims=tup), rval)
            @test isequal(vvmaximum(abs, A′, dims=tup), rval)
        end
    end

    @testset "+/-Inf in findmin/findmax/minimum/maximum" begin
        A = [Inf -Inf Inf  -Inf;
             Inf  Inf -Inf -Inf]
        A′ = [1 0 1 0;
              1 1 0 0]
        retinf(x::T) where {T} = ifelse(x == one(T), Inf, -Inf)
        for (tup, rval, rind) in [((1,), [Inf -Inf -Inf -Inf], [CartesianIndex(1,1) CartesianIndex(1,2) CartesianIndex(2,3) CartesianIndex(1,4)]),
                                  ((2,), reshape([-Inf -Inf], 2, 1), reshape([CartesianIndex(1,2),CartesianIndex(2,3)], 2, 1)),
                                  ((1,2), fill(-Inf,1,1),fill(CartesianIndex(1,2),1,1))]
            @test isequal(vfindmin1(A, dims=tup), (rval, rind))
            @test isequal(vfindmin1(retinf, A′, dims=tup), (rval, rind))
            @test isequal(vvminimum(A, dims=tup), rval)
            @test isequal(vvminimum(retinf, A′, dims=tup), rval)
        end

        for (tup, rval, rind) in [((1,), [Inf Inf Inf -Inf], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(1,3) CartesianIndex(1,4)]),
                                  ((2,), reshape([Inf Inf], 2, 1), reshape([CartesianIndex(1,1),CartesianIndex(2,1)], 2, 1)),
                                  ((1,2), fill(Inf,1,1),fill(CartesianIndex(1,1),1,1))]
            @test isequal(vfindmax1(A, dims=tup), (rval, rind))
            @test isequal(vfindmax1(retinf, A′, dims=tup), (rval, rind))
            @test isequal(vvmaximum(A, dims=tup), rval)
            @test isequal(vvmaximum(retinf, A′, dims=tup), rval)
        end
    end
end
