# Essentially, a direct reproduction of Base's reduce.jl tests.
# Some small allowances are made for deviations.

@test vvreduce(+, Int64[]) === Int64(0)
@test vvreduce(+, Int16[]) === 0 # here we promote to Int64
@test vvreduce(-, 1:5, init=0) == 15
@test vvreduce(-, 1:5; init=10) == 25

@test vvmapreduce((x)-> x ⊻ true, &, [true false true false false], init=true) == false
@test_throws UndefVarError vvmapreduce((x)-> x ⊻ true, |, [true false true false false]; init=false) == true

@test vvreduce(+, [1]) == 1 # Issue #21493

# reduce
@test vvreduce(max, [8 6 7 5 3 0 9]) == 9
@test vvreduce(+, 1:5; init=1000) == (1000 + 1 + 2 + 3 + 4 + 5)

# mapreduce
@test vvmapreduce(-, +, [-10 -9 -3]) == ((10 + 9) + 3)

# mapreduce with multiple iterators
@test vvmapreduce(*, +, 2:3, 4:5) == 23
@test vvmapreduce(*, +, 2:3, 4:5; init=2) == 25

@test vvmapreduce(*, +, [2, 3], [4, 5]) == 23
@test vvmapreduce(*, +, [2, 3], [4, 5]; init = 2) == 25
@test vvmapreduce(*, +, [2, 3], [4, 5]; dims = 1) == [23]
@test vvmapreduce(*, +, [2, 3], [4, 5]; dims = 1, init = 2) == [25]
@test vvmapreduce(*, +, [2, 3], [4, 5]; dims = 2) == [8, 15]
@test vvmapreduce(*, +, [2, 3], [4, 5]; dims = 2, init = 2) != [10, 17] # needs to be fixed

@test vvmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]) == 110
@test vvmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]; init = 2) == 112
@test vvmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]; dims = 1) == [44 66]
@test vvmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]; dims = 1, init = 2) == [46 68]
@test vvmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]; dims = 2) == reshape([33, 77], :, 1)
@test vvmapreduce(*, +, [2 3; 4 5], [6 7; 8 9]; dims = 2, init = 2) == reshape([35, 79], :, 1)

# mapreduce() for 1- 2- and n-sized blocks (PR #19325)
@test vvmapreduce(-, +, [-10]) == 10
@test vvmapreduce(abs2, +, [-9, -3]) == 81 + 9
@test vvmapreduce(-, +, [-9, -3, -4, 8, -2]) == (9 + 3 + 4 - 8 + 2)
@test vvmapreduce(-, +, Vector(range(1.0, stop=10000.0, length=10000))) == -50005000.0
# empty mr
@test vvmapreduce(abs2, +, Float64[]) === 0.0
@test vvmapreduce(abs2, *, Float64[]) === 1.0
@test vvmapreduce(abs2, max, Float64[]) === -Inf # base is 0.0
@test vvmapreduce(abs, max, Float64[]) === -Inf # base is 0.0

# mapreduce() type stability
@test typeof(vvmapreduce(*, +, Int8[10])) ===
    typeof(vvmapreduce(*, +, Int8[10, 11])) ===
    typeof(vvmapreduce(*, +, Int8[10, 11, 12, 13]))
@test typeof(vvmapreduce(*, +, Float32[10.0])) ===
    typeof(vvmapreduce(*, +, Float32[10, 11])) ===
    typeof(vvmapreduce(*, +, Float32[10, 11, 12, 13]))
@test typeof(vvmapreduce(abs, +, Int8[])) ===
    typeof(vvmapreduce(abs, +, Int8[10])) ===
    typeof(vvmapreduce(abs, +, Int8[10, 11])) ===
    typeof(vvmapreduce(abs, +, Int8[10, 11, 12, 13]))
@test typeof(vvmapreduce(abs, +, Float32[])) ===
    typeof(vvmapreduce(abs, +, Float32[10])) ===
    typeof(vvmapreduce(abs, +, Float32[10, 11])) ===
    typeof(vvmapreduce(abs, +, Float32[10, 11, 12, 13]))

# sum
@testset "vvsums promote to at least machine size" begin
    @testset for T in [Int8, Int16, Int32]
        @test vvsum(T[]) === Int(0)
    end
    # @testset for T in [UInt8, UInt16, UInt32]
    #     @test vvsum(T[]) === UInt(0)
    # end
    @testset for T in [Int, Int64, Int128, UInt, UInt64, UInt128,
                       Float32, Float64]
        @test vvsum(T[]) === T(0)
    end
    @test vvsum(BigInt[]) == big(0) && vvsum(BigInt[]) isa BigInt
end

@test vvsum(Bool[]) === vvsum(Bool[false]) === vvsum(Bool[false, false]) === 0
@test vvsum(Bool[true, false, true]) === 2

@test vvsum([Int8(3)]) === Int(3)
@test vvsum([3]) === 3
@test vvsum([3.0]) === 3.0

z = collect(reshape(1:16, (2,2,2,2)))
fz = float(z)
@test vvsum(z) === 136
@test vvsum(fz) === 136.0

@test vvsum(sin, Int[]) === 0.0
@test_throws MethodError vvsum(sin, 3) == sin(3.0)
@test vvsum(sin, [3]) == sin(3.0)
a = vvsum(sin, z)
@test a ≈ vvsum(sin, fz)
@test a ≈ vvsum(sin.(fz))

# prod
z = [-4, -3, 2, 5]
fz = float(z)
@test vvprod(Int[]) === 1
@test vvprod(Int8[]) === Int(1)
@test vvprod(Float64[]) === 1.0

@test vvprod([3]) === 3
@test vvprod([Int8(3)]) === Int(3)
@test vvprod([UInt(3)]) === UInt(3)
@test vvprod([3.0]) === 3.0

@test vvprod(z) === 120
@test vvprod(fz) === 120.0
@test vvprod(Array(trues(10))) == 1

# maximum & minimum & extrema
@test vvmaximum(Int[]) == typemin(Int)
@test vvminimum(Int[]) == typemax(Int)
@test vvextrema(Int[]) == (typemax(Int), typemin(Int))

@test vvmaximum(Int[]; init=-1) == -1
@test vvminimum(Int[]; init=-1) == -1
@test vvextrema(Int[]; init=(1, -1)) == (1, -1) # needs fixed

@test vvmaximum(sin, []; init=-1) == -1
@test vvminimum(sin, []; init=1) == 1
@test vvextrema(sin, []; init=(1, -1)) == (1, -1)

# @test vvmaximum(5) == 5
# @test vvminimum(5) == 5
# @test vvextrema(5) == (5, 5)
# @test vvextrema(abs2, 5) == (25, 25)

let x = [4,3,5,2]
    @test vvmaximum(x) == 5
    @test vvminimum(x) == 2
    @test vvextrema(x) == (2, 5)

    @test vvmaximum(abs2, x) == 25
    @test vvminimum(abs2, x) == 4
    @test vvextrema(abs2, x) == (4, 25)
end

@test vvmaximum([-0.,0.]) === 0.0
@test vvmaximum([0.,-0.]) === -0.0 # opposite from base
@test vvmaximum([0.,-0.,0.]) === -0.0 # opposite from base
@test vvminimum([-0.,0.]) === 0.0 # opposite from base
@test vvminimum([0.,-0.]) === -0.0
@test vvminimum([0.,-0.,0.]) === -0.0

@testset "minimum/maximum checks all elements" begin
    for N in [2:20;150;300]
        for i in 1:N
            arr = fill(0., N)
            truth = rand()
            arr[i] = truth
            @test vvmaximum(arr) == truth

            truth = -rand()
            arr[i] = truth
            @test vvminimum(arr) == truth

            arr[i] = NaN
            @test !isnan(vvmaximum(arr)) # NaN not handled
            @test !isnan(vvminimum(arr)) # NaN not handled

            arr = zeros(N)
            @test vvminimum(arr) === 0.0
            @test vvmaximum(arr) === 0.0

            # arr[i] = -0.0
            # @test vvminimum(arr) === 0.0 # opposite from base
            # @test vvmaximum(arr) ===  0.0

            arr = -zeros(N)
            @test vvminimum(arr) === -0.0
            @test vvmaximum(arr) === -0.0
            # arr[i] = 0.0
            # @test vvminimum(arr) === -0.0
            # @test vvmaximum(arr) === -0.0 # opposite from base
        end
    end
end

@testset "maximum works on generic order #30320" begin
    for n in [1:20;1500]
        arr = randn(n)
        @test GenericOrder(maximum(arr)) === maximum(map(GenericOrder, arr))
        @test GenericOrder(minimum(arr)) === minimum(map(GenericOrder, arr))
        f = x -> x
        @test GenericOrder(maximum(f,arr)) === maximum(f,map(GenericOrder, arr))
        @test GenericOrder(minimum(f,arr)) === minimum(f,map(GenericOrder, arr))
    end
end

@testset "maximum no out of bounds access #30462" begin
    arr = fill(-Inf, 128,128)
    @test vvmaximum(arr) == -Inf
    arr = fill(Inf, 128^2)
    @test vvminimum(arr) == Inf
    for center in [256, 1024, 4096, 128^2]
        for offset in -10:10
            len = center + offset
            x = randn()
            arr = fill(x, len)
            @test vvmaximum(arr) === x
            @test vvminimum(arr) === x
        end
    end
end

# NaN handling differs from base, but we must still test to ensure it is consistent
@test isinf(vvmaximum([NaN]))
@test isinf(vvminimum([NaN]))
@test isequal(vvextrema([NaN]), (Inf, -Inf))

@test !isnan(vvmaximum([NaN, 2.]))
@test !isnan(vvmaximum([2., NaN]))
@test !isnan(vvminimum([NaN, 2.]))
@test !isnan(vvminimum([2., NaN]))
@test isequal(vvextrema([NaN, 2.]), (2., 2.))

@test !isnan(vvmaximum([NaN, 2., 3.]))
@test !isnan(vvminimum([NaN, 2., 3.]))
@test isequal(vvextrema([NaN, 2., 3.]), (2., 3.))

@test !isnan(vvmaximum([4., 3., NaN, 5., 2.]))
@test !isnan(vvminimum([4., 3., NaN, 5., 2.]))
@test isequal(vvextrema([4., 3., NaN, 5., 2.]), (2., 5.))

# test long arrays
@test !isnan(vvmaximum([NaN; 1.:10000.]))
@test !isnan(vvmaximum([1.:10000.; NaN]))
@test !isnan(vvminimum([NaN; 1.:10000.]))
@test !isnan(vvminimum([1.:10000.; NaN]))
@test isequal(vvextrema([1.:10000.; NaN]), (1., 10000.))
@test isequal(vvextrema([NaN; 1.:10000.]), (1., 10000.))

@test vvmaximum(abs2, 3:7) == 49
@test vvminimum(abs2, 3:7) == 9
@test vvextrema(abs2, 3:7) == (9, 49)

# here we promote
@test vvmaximum(Int16[1]) === 1
@test vvmaximum(Vector(Int16(1):Int16(100))) === 100
@test vvmaximum(Int32[1,2]) === 2

A = circshift(reshape(1:24,2,3,4), (0,1,1))
@test vvextrema(A,dims=1) == reshape([(23,24),(19,20),(21,22),(5,6),(1,2),(3,4),(11,12),(7,8),(9,10),(17,18),(13,14),(15,16)],1,3,4)
@test vvextrema(A,dims=2) == reshape([(19,23),(20,24),(1,5),(2,6),(7,11),(8,12),(13,17),(14,18)],2,1,4)
@test vvextrema(A,dims=3) == reshape([(5,23),(6,24),(1,19),(2,20),(3,21),(4,22)],2,3,1)
@test vvextrema(A,dims=(1,2)) == reshape([(19,24),(1,6),(7,12),(13,18)],1,1,4)
@test vvextrema(A,dims=(1,3)) == reshape([(5,24),(1,20),(3,22)],1,3,1)
@test vvextrema(A,dims=(2,3)) == reshape([(1,23),(2,24)],2,1,1)
@test vvextrema(A,dims=(1,2,3)) == reshape([(1,24)],1,1,1)
@test size(vvextrema(A,dims=1)) == size(maximum(A,dims=1))
@test size(vvextrema(A,dims=(1,2))) == size(maximum(A,dims=(1,2)))
@test size(vvextrema(A,dims=(1,2,3))) == size(maximum(A,dims=(1,2,3)))
@test vvextrema(x->div(x, 2), A, dims=(2,3)) == reshape([(0,11),(1,12)],2,1,1)


# any & all

# @test @inferred vany([]) == false
@test @inferred vany(Bool[]) == false
@test @inferred vany([true]) == true
@test @inferred vany([false, false]) == false
@test @inferred vany([false, true]) == true
@test @inferred vany([true, false]) == true
@test @inferred vany([true, true]) == true
@test @inferred vany([true, true, true]) == true
@test @inferred vany([true, false, true]) == true
@test @inferred vany([false, false, false]) == false

# @test @inferred vall([]) == true
@test @inferred vall(Bool[]) == true
@test @inferred vall([true]) == true
@test @inferred vall([false, false]) == false
@test @inferred vall([false, true]) == false
@test @inferred vall([true, false]) == false
@test @inferred vall([true, true]) == true
@test @inferred vall([true, true, true]) == true
@test @inferred vall([true, false, true]) == false
@test @inferred vall([false, false, false]) == false

# @test @inferred vany(x->x>0, []) == false
@test @inferred vany(x->x>0, Int[]) == false
@test @inferred vany(x->x>0, [-3]) == false
@test @inferred vany(x->x>0, [4]) == true
@test @inferred vany(x->x>0, [-3, 4, 5]) == true

# @test @inferred vall(x->x>0, []) == true
@test @inferred vall(x->x>0, Int[]) == true
@test @inferred vall(x->x>0, [-3]) == false
@test @inferred vall(x->x>0, [4]) == true
@test @inferred vall(x->x>0, [-3, 4, 5]) == false

let f(x) = ifelse(x == 1, true, ifelse(x == 2, false, 1))
    @test vany(Any[false,true,false])
    @test @inferred vany(map(f, [2,1,2]))
    @test @inferred vany([f(x) for x in [2,1,2]])

    @test vall(Any[true,true,true])
    @test @inferred vall(map(f, [1,1,1]))
    @test @inferred vall([f(x) for x in [1,1,1]])

    # @test_throws TypeError vany([1,true])
    # @test_throws TypeError vall([true,1])
    # @test_throws TypeError vany(map(f,[3,1]))
    # @test_throws TypeError vall(map(f,[1,3]))
end


@test vcount(x -> x > 0, Int[]) == vcount(Bool[]) == 0
@test vtcount(x -> x > 0, Int[]) == vtcount(Bool[]) == 0
@test vcount(x->x>0, -3:5) == vcount((-3:5) .> 0) == 5
@test vcount([true, true, false, true]) == vcount(BitVector([true, true, false, true])) == 3
x = repeat([false, true, false, true, true, false], 7)
@test vcount(x) == 21
@test_throws MethodError vcount(sqrt, [1])
@test_throws MethodError vcount([1])
@test vcount(!iszero, Int[]) == 0
@test vcount(!iszero, Int[0]) == 0
@test vcount(!iszero, Int[1]) == 1
@test vcount(!iszero, [1, 0, 2, 0, 3, 0, 4]) == 4

@test vvsum(Vector(map(UInt8,0:255))) == 32640
@test vvsum(Vector(map(UInt8,254:255))) == 509

# opposite behavior from base
@test vvsum([-0.0]) === 0.0
@test vvsum([-0.0, -0.0]) === 0.0
# same as base
@test vvprod([-0.0, -0.0]) === 0.0
