# Essentially, a direct reproduction of Base's reducedim.jl tests.
# Some small allowances are made for deviations.
Random.seed!(0x537335fdb9e832b1)

@testset "test reductions over region: $region" for region in Any[
    1, 2, 3, 4, 5, (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
    (1, 2, 3), (1, 3, 4), (2, 3, 4), (1, 2, 3, 4), :]
    A = rand(3, 4, 5, 6)

    @test sum(A, dims=region) ≈ vvsum(A, dims=region)
    @test prod(A, dims=region) ≈ vvprod(A, dims=region)
    @test maximum(A, dims=region) ≈ vvmaximum(A, dims=region)
    @test minimum(A, dims=region) ≈ vvminimum(A, dims=region)

    @test sum(abs2, A, dims=region) ≈ vvsum(abs2, A, dims=region)
    @test prod(abs2, A, dims=region) ≈ vvprod(abs2, A, dims=region)
    @test maximum(abs2, A, dims=region) ≈ vvmaximum(abs2, A, dims=region)
    @test minimum(abs2, A, dims=region) ≈ vvminimum(abs2, A, dims=region)


    @test count(≥(0.5), A, dims=region) == vcount(≥(0.5), A, dims=region)

    # With init=false
    @test sum(A, init=false) ≈ vvsum(A, init=false)
    @test prod(A, init=false) ≈ vvprod(A, init=false)
    @test maximum(A, init=false) ≈ vvmaximum(A, init=false)
    @test minimum(A, init=false) ≈ vvminimum(A, init=false)

    @test sum(abs2, A, init=false) ≈ vvsum(abs2, A, init=false)
    @test prod(abs2, A, init=false) ≈ vvprod(abs2, A, init=false)
    @test maximum(abs2, A, init=false) ≈ vvmaximum(abs2, A, init=false)
    @test minimum(abs2, A, init=false) ≈ vvminimum(abs2, A, init=false)

    @test @inferred vvsum(A, dims=region) ≈ sum(A, dims=region)
    @test @inferred(vvprod(A, dims=region)) ≈ prod(A, dims=region)
    @test @inferred(vvmaximum(A, dims=region)) ≈ maximum(A, dims=region)
    @test @inferred(vvminimum(A, dims=region)) ≈ minimum(A, dims=region)

    @test @inferred(vvsum(abs, A, dims=region)) ≈ sum(abs, A, dims=region)
    @test @inferred(vvsum(abs2, A, dims=region)) ≈ sum(abs2, A, dims=region)
    @test @inferred(vvprod(abs2, A, dims=region)) ≈ prod(abs2, A, dims=region)
    @test @inferred(vvmaximum(abs, A, dims=region)) ≈ maximum(abs, A, dims=region)
    @test @inferred(vvminimum(abs, A, dims=region)) ≈ minimum(abs, A, dims=region)

    # With numeric init
    @test vvextrema(A, dims=region, init=(.1, .3)) == extrema(A, dims=region, init=(.1, .3))
    @test vvextrema(abs2, A, dims=region, init=(.1, .3)) == extrema(abs2, A, dims=region, init=(.1, .3))
    # With mixed init
    @test vvextrema(A, dims=region, init=(typemin, .3)) == extrema(A, dims=region, init=(typemin(Float64), .3))
    @test vvextrema(abs2, A, dims=region, init=(typemin, .3)) == extrema(abs2, A, dims=region, init=(typemin(Float64), .3))
    @test vvextrema(A, dims=region, init=(3., zero)) == extrema(A, dims=region, init=(3., 0.))
    @test vvextrema(abs2, A, dims=region, init=(3., zero)) == extrema(abs2, A, dims=region, init=(3., 0.))
end

# Small integers
@test @inferred(vvsum(Int8[1], dims=1)) == [1]
@test @inferred(vvsum(UInt8[1], dims=1)) == [1]

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

@testset "NaN in findmin/findmax" begin
    A = [1.0 NaN 6.0;
         NaN 2.0 4.0]
    A′ = [-1.0 NaN -6.0;
          NaN -2.0 4.0]
    for (tup, rval, rind) in [((1,), [1.0 2.0 4.0], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(2,3)]),
                              ((2,), reshape([1.0, 2.0], 2, 1), reshape([CartesianIndex(1,1),CartesianIndex(2,2)], 2, 1)),
                              ((1,2), fill(1.0,1,1),fill(CartesianIndex(1,1),1,1))]
        @test isequal(vfindmin1(A, dims=tup), (rval, rind))
        @test isequal(vfindmin1(abs, A′, dims=tup), (rval, rind))
    end

    for (tup, rval, rind) in [((1,), [1.0 2.0 6.0], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(1,3)]),
                              ((2,), reshape([6.0, 4.0], 2, 1), reshape([CartesianIndex(1,3),CartesianIndex(2,3)], 2, 1)),
                              ((1,2), fill(6.0,1,1),fill(CartesianIndex(1,3),1,1))]
        @test isequal(vfindmax1(A, dims=tup), (rval, rind))
        @test isequal(vfindmax1(abs, A′, dims=tup), (rval, rind))
    end
end

@testset "NaN in minimum/maximum" begin
    A = [1.0 NaN 6.0;
         NaN 2.0 4.0]
    A′ = [-1.0 NaN -6.0;
          NaN -2.0 4.0]
    for (tup, rval, rind) in [((1,), [NaN 2.0 4.0], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(2,3)]),
                              ((2,), reshape([6.0, 2.0], 2, 1), reshape([CartesianIndex(1,1),CartesianIndex(2,2)], 2, 1)),
                              ((1,2), fill(1.0,1,1),fill(CartesianIndex(1,1),1,1))]
        @test isequal(vvminimum(A, dims=tup), rval)
        @test isequal(vvminimum(abs, A′, dims=tup), rval)
    end

    for (tup, rval, rind) in [((1,), [NaN 2.0 6.0], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(1,3)]),
                              ((2,), reshape([6.0, 4.0], 2, 1), reshape([CartesianIndex(1,3),CartesianIndex(2,3)], 2, 1)),
                              ((1,2), fill(6.0,1,1),fill(CartesianIndex(1,3),1,1))]
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

@testset "region=$region" for region in Any[[0, 1], [0 1; 2 3], "hello"]
    Areduc = rand(3, 4, 5, 6)

    @test_throws MethodError vvsum(Areduc, dims=region)
    @test_throws MethodError vvprod(Areduc, dims=region)
    @test_throws MethodError vvmaximum(Areduc, dims=region)
    @test_throws MethodError vvminimum(Areduc, dims=region)
    @test_throws MethodError vvsum(abs, Areduc, dims=region)
    @test_throws MethodError vvsum(abs2, Areduc, dims=region)
    @test_throws MethodError vvmaximum(abs, Areduc, dims=region)
    @test_throws MethodError vvminimum(abs, Areduc, dims=region)
end
