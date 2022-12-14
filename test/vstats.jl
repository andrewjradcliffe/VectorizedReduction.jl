counteq(x, y; dims=:) = mapreduce(==, +, x, y, dims=dims)
countne(x, y; dims=:) = mapreduce(!=, +, x, y, dims=dims)
@testset "counteq" begin
    for T ∈ (Float32, Float64, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)
        x, y = rand(T, 1000), rand(T, 1000)
        c = counteq(x, y)
        @test vcounteq(x, y) == c
        @test vtcounteq(x, y) == c
    end
    x, y = rand(1:100, 10, 10, 10, 10), rand(1:100, 10, 10, 10, 10)
    for dims ∈ (1, 2, 3, 4, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), (1,2,3,4,5))
        c = counteq(x, y, dims=dims)
        @test vcounteq(x, y, dims=dims) == c
        @test vtcounteq(x, y, dims=dims) == c
    end
end

@testset "countne" begin
    for T ∈ (Float32, Float64, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)
        x, y = rand(T, 1000), rand(T, 1000)
        c = countne(x, y)
        @test vcountne(x, y) == c
        @test vtcountne(x, y) == c
    end
    x, y = rand(1:100, 10, 10, 10, 10), rand(1:100, 10, 10, 10, 10)
    for dims ∈ (1, 2, 3, 4, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), (1,2,3,4,5))
        c = countne(x, y, dims=dims)
        @test vcountne(x, y, dims=dims) == c
        @test vtcountne(x, y, dims=dims) == c
    end
end

meanad(x, y; dims=:) = mean(abs.(x .- y), dims=dims)
maxad(x, y; dims=:) = maximum(abs.(x .- y), dims=dims)
mse(x, y; dims=:) = mean(abs2.(x .- y), dims=dims)
rmse(x, y; dims=:) = .√mean(abs2.(x .- y), dims=dims)

@testset "meanad" begin
    for T ∈ (Float32, Float64, Int32)
        x, y = rand(T, 1000), rand(T, 1000)
        c = meanad(x, y)
        @test vmeanad(x, y) ≈ c
        @test vtmeanad(x, y) ≈ c
    end
    x, y = rand(10, 10, 10, 10), rand(10, 10, 10, 10)
    for dims ∈ (1, 2, 3, 4, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), (1,2,3,4,5))
        c = meanad(x, y, dims=dims)
        @test vmeanad(x, y, dims=dims) ≈ c
        @test vtmeanad(x, y, dims=dims) ≈ c
    end
end

@testset "maxad" begin
    for T ∈ (Float32, Float64, Int32)
        x, y = rand(T, 1000), rand(T, 1000)
        c = maxad(x, y)
        @test vmaxad(x, y) ≈ c
        @test vtmaxad(x, y) ≈ c
    end
    x, y = rand(10, 10, 10, 10), rand(10, 10, 10, 10)
    for dims ∈ (1, 2, 3, 4, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), (1,2,3,4,5))
        c = maxad(x, y, dims=dims)
        @test vmaxad(x, y, dims=dims) ≈ c
        @test vtmaxad(x, y, dims=dims) ≈ c
    end
end

@testset "mse" begin
    for T ∈ (Float32, Float64, Int32)
        x, y = rand(T, 1000), rand(T, 1000)
        c = mse(x, y)
        @test vmse(x, y) ≈ c
        @test vtmse(x, y) ≈ c
    end
    x, y = rand(10, 10, 10, 10), rand(10, 10, 10, 10)
    for dims ∈ (1, 2, 3, 4, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), (1,2,3,4,5))
        c = mse(x, y, dims=dims)
        @test vmse(x, y, dims=dims) ≈ c
        @test vtmse(x, y, dims=dims) ≈ c
    end
end

@testset "rmse" begin
    for T ∈ (Float32, Float64, Int32)
        x, y = rand(T, 1000), rand(T, 1000)
        c = rmse(x, y)
        @test vrmse(x, y) ≈ c
        @test vtrmse(x, y) ≈ c
    end
    x, y = rand(10, 10, 10, 10), rand(10, 10, 10, 10)
    for dims ∈ (1, 2, 3, 4, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), (1,2,3,4,5))
        c = rmse(x, y, dims=dims)
        @test vrmse(x, y, dims=dims) ≈ c
        @test vtrmse(x, y, dims=dims) ≈ c
    end
end
