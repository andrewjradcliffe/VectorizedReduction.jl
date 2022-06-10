# Essentially, a direct reproduction of Base's reduce.jl tests.
# Some small allowances are made for deviations.

@test vtreduce(+, Int64[]) === Int64(0)
@test vtreduce(+, Int16[]) === 0 # here we promote to Int64
@test vtreduce(-, 1:5, init=0) == 15
@test vtreduce(-, 1:5; init=10) == 25

@test vtmapreduce((x)-> x ⊻ true, &, [true false true false false], init=true) == false
@test_throws UndefVarError vtmapreduce((x)-> x ⊻ true, |, [true false true false false]; init=false) == true

@test vtreduce(+, [1]) == 1 # Issue #21493

# reduce
@test vtreduce(max, [8 6 7 5 3 0 9]) == 9
@test vtreduce(+, 1:5; init=1000) == (1000 + 1 + 2 + 3 + 4 + 5)

# mapreduce
@test vtmapreduce(-, +, [-10 -9 -3]) == ((10 + 9) + 3)

# mapreduce with multiple iterators
@test vtmapreduce(*, +, 2:3, 4:5) == 23
@test vtmapreduce(*, +, 2:3, 4:5; init=2) == 25

@test vtmapreduce(*, +, [2, 3], [4, 5]) == 23
@test vtmapreduce(*, +, [2, 3], [4, 5]; init = 2) == 25
@test vtmapreduce(*, +, [2, 3], [4, 5]; dims = 1) == [23]
@test vtmapreduce(*, +, [2, 3], [4, 5]; dims = 1, init = 2) == [25]
@test vtmapreduce(*, +, [2, 3], [4, 5]; dims = 2) == [8, 15]
@test vtmapreduce(*, +, [2, 3], [4, 5]; dims = 2, init = 2) != [10, 17] # needs to be fixed

@test vtmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]) == 110
@test vtmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]; init = 2) == 112
@test vtmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]; dims = 1) == [44 66]
@test vtmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]; dims = 1, init = 2) == [46 68]
@test vtmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]; dims = 2) == reshape([33, 77], :, 1)
@test vtmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]; dims = 2, init = 2) == reshape([35, 79], :, 1)

# mapreduce() for 1- 2- and n-sized blocks (PR #19325)
@test vtmapreduce(-, +, [-10]) == 10
@test vtmapreduce(abs2, +, [-9, -3]) == 81 + 9
@test vtmapreduce(-, +, [-9, -3, -4, 8, -2]) == (9 + 3 + 4 - 8 + 2)
@test vtmapreduce(-, +, Vector(range(1.0, stop=10000.0, length=10000))) == -50005000.0
# empty mr
@test vtmapreduce(abs2, +, Float64[]) === 0.0
@test vtmapreduce(abs2, *, Float64[]) === 1.0
@test vtmapreduce(abs2, max, Float64[]) === -Inf # base is 0.0
@test vtmapreduce(abs, max, Float64[]) === -Inf # base is 0.0

# mapreduce() type stability
@test typeof(vtmapreduce(*, +, Int8[10])) ===
    typeof(vtmapreduce(*, +, Int8[10, 11])) ===
    typeof(vtmapreduce(*, +, Int8[10, 11, 12, 13]))
@test typeof(vtmapreduce(*, +, Float32[10.0])) ===
    typeof(vtmapreduce(*, +, Float32[10, 11])) ===
    typeof(vtmapreduce(*, +, Float32[10, 11, 12, 13]))
@test typeof(vtmapreduce(abs, +, Int8[])) ===
    typeof(vtmapreduce(abs, +, Int8[10])) ===
    typeof(vtmapreduce(abs, +, Int8[10, 11])) ===
    typeof(vtmapreduce(abs, +, Int8[10, 11, 12, 13]))
@test typeof(vtmapreduce(abs, +, Float32[])) ===
    typeof(vtmapreduce(abs, +, Float32[10])) ===
    typeof(vtmapreduce(abs, +, Float32[10, 11])) ===
    typeof(vtmapreduce(abs, +, Float32[10, 11, 12, 13]))

# sum
@testset "vtsums promote to at least machine size" begin
    @testset for T in [Int8, Int16, Int32]
        @test vtsum(T[]) === Int(0)
    end
    # @testset for T in [UInt8, UInt16, UInt32]
    #     @test vtsum(T[]) === UInt(0)
    # end
    @testset for T in [Int, Int64, UInt, UInt64, Float32, Float64]
        @test vtsum(T[]) === T(0)
    end
end

@test vtsum(Bool[]) === vtsum(Bool[false]) === vtsum(Bool[false, false]) === 0
@test vtsum(Bool[true, false, true]) === 2

@test vtsum([Int8(3)]) === Int(3)
@test vtsum([3]) === 3
@test vtsum([3.0]) === 3.0

z = collect(reshape(1:16, (2,2,2,2)))
fz = float(z)
@test vtsum(z) === 136
@test vtsum(fz) === 136.0

@test vtsum(sin, Int[]) === 0.0
@test_throws MethodError vtsum(sin, 3) == sin(3.0)
@test vtsum(sin, [3]) == sin(3.0)
a = vtsum(sin, z)
@test a ≈ vtsum(sin, fz)
@test a ≈ vtsum(sin.(fz))

# prod
z = [-4, -3, 2, 5]
fz = float(z)
@test vtprod(Int[]) === 1
@test vtprod(Int8[]) === Int(1)
@test vtprod(Float64[]) === 1.0

@test vtprod([3]) === 3
@test vtprod([Int8(3)]) === Int(3)
@test vtprod([UInt(3)]) === UInt(3)
@test vtprod([3.0]) === 3.0

@test vtprod(z) === 120
@test vtprod(fz) === 120.0
@test vtprod(Array(trues(10))) == 1

# maximum & minimum & extrema
@test vtmaximum(Int[]) == typemin(Int)
@test vtminimum(Int[]) == typemax(Int)
@test vtextrema(Int[]) == (typemax(Int), typemin(Int))

@test vtmaximum(Int[]; init=-1) == -1
@test vtminimum(Int[]; init=-1) == -1
@test vtextrema(Int[]; init=(1, -1)) == (1, -1) # needs fixed

@test vtmaximum(sin, Int[]; init=-1) == -1
@test vtminimum(sin, Int[]; init=1) == 1
@test vtextrema(sin, Int[]; init=(1, -1)) == (1, -1)

# @test vtmaximum(5) == 5
# @test vtminimum(5) == 5
# @test vtextrema(5) == (5, 5)
# @test vtextrema(abs2, 5) == (25, 25)

let x = [4,3,5,2]
    @test vtmaximum(x) == 5
    @test vtminimum(x) == 2
    @test vtextrema(x) == (2, 5)

    @test vtmaximum(abs2, x) == 25
    @test vtminimum(abs2, x) == 4
    @test vtextrema(abs2, x) == (4, 25)
end

@test vtmaximum([-0.,0.]) === 0.0
@test vtmaximum([0.,-0.]) === -0.0 # opposite from base
@test vtmaximum([0.,-0.,0.]) === -0.0 # opposite from base
@test vtminimum([-0.,0.]) === 0.0 # opposite from base
@test vtminimum([0.,-0.]) === -0.0
@test vtminimum([0.,-0.,0.]) === -0.0

@testset "minimum/maximum/extrema checks all elements" begin
    for N in [2:20;150;300]
        for i in 1:N
            arr = fill(0., N)
            truth = rand()
            arr[i] = truth
            @test vtmaximum(arr) == truth
            @test vtextrema(arr) == (0., truth)

            truth = -rand()
            arr[i] = truth
            @test vtminimum(arr) == truth
            @test vtextrema(arr) == (truth, 0.)

            arr[i] = NaN
            @test !isnan(vtmaximum(arr)) # NaN not handled
            @test !isnan(vtminimum(arr)) # NaN not handled
            @test all(!isnan, vtextrema(arr))

            arr = zeros(N)
            @test vtminimum(arr) === 0.0
            @test vtmaximum(arr) === 0.0
            @test vtextrema(arr) === (0.0, 0.0)

            # arr[i] = -0.0
            # @test vtminimum(arr) === 0.0 # opposite from base
            # @test vtmaximum(arr) ===  0.0

            arr = -zeros(N)
            @test vtminimum(arr) === -0.0
            @test vtmaximum(arr) === -0.0
            @test vvextrema(arr) === (-0.0, -0.0)
            # arr[i] = 0.0
            # @test vtminimum(arr) === -0.0
            # @test vtmaximum(arr) === -0.0 # opposite from base
        end
    end
end

@testset "maximum no out of bounds access #30462" begin
    arr = fill(-Inf, 128,128)
    @test vtmaximum(arr) == -Inf
    @test vtextrema(arr) == (-Inf, -Inf)
    arr = fill(Inf, 128^2)
    @test vtminimum(arr) == Inf
    @test vtextrema(arr) == (Inf, Inf)
    for center in [256, 1024, 4096, 128^2]
        for offset in -10:10
            len = center + offset
            x = randn()
            arr = fill(x, len)
            @test vtmaximum(arr) === x
            @test vtminimum(arr) === x
            @test vtextrema(arr) === (x, x)
        end
    end
end

# NaN handling differs from base, but we must still test to ensure it is consistent
@test isinf(vtmaximum([NaN]))
@test isinf(vtminimum([NaN]))
@test isequal(vtextrema([NaN]), (Inf, -Inf))

@test !isnan(vtmaximum([NaN, 2.]))
@test !isnan(vtmaximum([2., NaN]))
@test !isnan(vtminimum([NaN, 2.]))
@test !isnan(vtminimum([2., NaN]))
@test isequal(vtextrema([NaN, 2.]), (2., 2.))

@test !isnan(vtmaximum([NaN, 2., 3.]))
@test !isnan(vtminimum([NaN, 2., 3.]))
@test isequal(vtextrema([NaN, 2., 3.]), (2., 3.))

@test !isnan(vtmaximum([4., 3., NaN, 5., 2.]))
@test !isnan(vtminimum([4., 3., NaN, 5., 2.]))
@test isequal(vtextrema([4., 3., NaN, 5., 2.]), (2., 5.))

# test long arrays
@test !isnan(vtmaximum([NaN; 1.:10000.]))
@test !isnan(vtmaximum([1.:10000.; NaN]))
@test !isnan(vtminimum([NaN; 1.:10000.]))
@test !isnan(vtminimum([1.:10000.; NaN]))
# @test isequal(vtextrema([1.:10000.; NaN]), (1., 10000.))
@test isequal(vtextrema([NaN; 1.:10000.]), (1., 10000.))

@test vtmaximum(abs2, 3:7) == 49
@test vtminimum(abs2, 3:7) == 9
@test vtextrema(abs2, 3:7) == (9, 49)

# here we promote
@test vtmaximum(Int16[1]) === 1
@test vtmaximum(Vector(Int16(1):Int16(100))) === 100
@test vtmaximum(Int32[1,2]) === 2

A = circshift(reshape(1:24,2,3,4), (0,1,1))
@test vtextrema(A,dims=1) == reshape([(23,24),(19,20),(21,22),(5,6),(1,2),(3,4),(11,12),(7,8),(9,10),(17,18),(13,14),(15,16)],1,3,4)
@test vtextrema(A,dims=2) == reshape([(19,23),(20,24),(1,5),(2,6),(7,11),(8,12),(13,17),(14,18)],2,1,4)
@test vtextrema(A,dims=3) == reshape([(5,23),(6,24),(1,19),(2,20),(3,21),(4,22)],2,3,1)
@test vtextrema(A,dims=(1,2)) == reshape([(19,24),(1,6),(7,12),(13,18)],1,1,4)
@test vtextrema(A,dims=(1,3)) == reshape([(5,24),(1,20),(3,22)],1,3,1)
@test vtextrema(A,dims=(2,3)) == reshape([(1,23),(2,24)],2,1,1)
@test vtextrema(A,dims=(1,2,3)) == reshape([(1,24)],1,1,1)
@test size(vtextrema(A,dims=1)) == size(maximum(A,dims=1))
@test size(vtextrema(A,dims=(1,2))) == size(maximum(A,dims=(1,2)))
@test size(vtextrema(A,dims=(1,2,3))) == size(maximum(A,dims=(1,2,3)))
@test vtextrema(x->div(x, 2), A, dims=(2,3)) == reshape([(0,11),(1,12)],2,1,1)


# any & all

# @test @inferred vtany([]) == false
@test @inferred vtany(Bool[]) == false
@test @inferred vtany([true]) == true
@test @inferred vtany([false, false]) == false
@test @inferred vtany([false, true]) == true
@test @inferred vtany([true, false]) == true
@test @inferred vtany([true, true]) == true
@test @inferred vtany([true, true, true]) == true
@test @inferred vtany([true, false, true]) == true
@test @inferred vtany([false, false, false]) == false

# @test @inferred vtall([]) == true
@test @inferred vtall(Bool[]) == true
@test @inferred vtall([true]) == true
@test @inferred vtall([false, false]) == false
@test @inferred vtall([false, true]) == false
@test @inferred vtall([true, false]) == false
@test @inferred vtall([true, true]) == true
@test @inferred vtall([true, true, true]) == true
@test @inferred vtall([true, false, true]) == false
@test @inferred vtall([false, false, false]) == false

# @test @inferred vtany(x->x>0, []) == false
@test @inferred vtany(x->x>0, Int[]) == false
@test @inferred vtany(x->x>0, [-3]) == false
@test @inferred vtany(x->x>0, [4]) == true
@test @inferred vtany(x->x>0, [-3, 4, 5]) == true

# @test @inferred vtall(x->x>0, []) == true
@test @inferred vtall(x->x>0, Int[]) == true
@test @inferred vtall(x->x>0, [-3]) == false
@test @inferred vtall(x->x>0, [4]) == true
@test @inferred vtall(x->x>0, [-3, 4, 5]) == false

# let f(x) = ifelse(x == 1, true, ifelse(x == 2, false, 1))
#     @test vtany(Any[false,true,false])
#     @test @inferred vtany(map(f, [2,1,2]))
#     @test @inferred vtany([f(x) for x in [2,1,2]])

#     @test vtall(Any[true,true,true])
#     @test @inferred vtall(map(f, [1,1,1]))
#     @test @inferred vtall([f(x) for x in [1,1,1]])

#     # @test_throws TypeError vtany([1,true])
#     # @test_throws TypeError vtall([true,1])
#     # @test_throws TypeError vtany(map(f,[3,1]))
#     # @test_throws TypeError vtall(map(f,[1,3]))
# end


@test vtcount(x -> x > 0, Int[]) == vtcount(Bool[]) == 0
@test vtcount(x -> x > 0, Int[]) == vtcount(Bool[]) == 0
@test vtcount(x->x>0, -3:5) == vtcount((-3:5) .> 0) == 5
@test vtcount([true, true, false, true]) == vtcount(BitVector([true, true, false, true])) == 3
x = repeat([false, true, false, true, true, false], 7)
@test vtcount(x) == 21
@test_throws MethodError vtcount(sqrt, [1])
@test_throws MethodError vtcount([1])
@test vtcount(!iszero, Int[]) == 0
@test vtcount(!iszero, Int[0]) == 0
@test vtcount(!iszero, Int[1]) == 1
@test vtcount(!iszero, [1, 0, 2, 0, 3, 0, 4]) == 4

@test vtsum(Vector(map(UInt8,0:255))) == 32640
@test vtsum(Vector(map(UInt8,254:255))) == 509

# opposite behavior from base
@test vtsum([-0.0]) === 0.0
@test vtsum([-0.0, -0.0]) === 0.0
# same as base
@test vtprod([-0.0, -0.0]) === 0.0
