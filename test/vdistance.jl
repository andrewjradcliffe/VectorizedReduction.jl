Random.seed!(0x3c021fc14523fc9a)
@testset "1-dimensional distances" begin
    x, y = rand(1000), rand(1000)
    @test vmanhattan(x, y) ≈ mapreduce(abs ∘ -, +, x, y)
    @test vtmanhattan(x, y) ≈ mapreduce(abs ∘ -, +, x, y)
    @test veuclidean(x, y) ≈ √mapreduce(abs2 ∘ -, +, x, y)
    @test vteuclidean(x, y) ≈ √mapreduce(abs2 ∘ -, +, x, y)
    @test vchebyshev(x, y) ≈ mapreduce(abs ∘ -, max, x, y)
    @test vtchebyshev(x, y) ≈ mapreduce(abs ∘ -, max, x, y)
    for p ∈ (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        d = mapreduce((a, b) -> abs(a - b)^p, +, x, y)^(1/p)
        @test vminkowski(x, y, p) ≈ d
        @test vtminkowski(x, y, p) ≈ d
    end
    @test vminkowski(x, y, Inf) ≈ mapreduce(abs ∘ -, max, x, y)
    @test vminkowski(x, y, -Inf) ≈ mapreduce(abs ∘ -, min, x, y)
    @test vtminkowski(x, y, Inf) ≈ mapreduce(abs ∘ -, max, x, y)
    @test vtminkowski(x, y, -Inf) ≈ mapreduce(abs ∘ -, min, x, y)
end

@testset "multi-dimensional distances" begin
    x, y = rand(10,10,10,10), rand(10,10,10,10)
    for dims ∈ (1, 2, 3, 4, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4))
        @test vmanhattan(x, y, dims=dims) ≈ mapreduce(abs ∘ -, +, x, y, dims=dims)
        @test vtmanhattan(x, y, dims=dims) ≈ mapreduce(abs ∘ -, +, x, y, dims=dims)
        @test veuclidean(x, y, dims=dims) ≈ .√mapreduce(abs2 ∘ -, +, x, y, dims=dims)
        @test vteuclidean(x, y, dims=dims) ≈ .√mapreduce(abs2 ∘ -, +, x, y, dims=dims)
        @test vchebyshev(x, y, dims=dims) ≈ mapreduce(abs ∘ -, max, x, y, dims=dims)
        @test vtchebyshev(x, y, dims=dims) ≈ mapreduce(abs ∘ -, max, x, y, dims=dims)
        for p ∈ (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
            d = mapreduce((a, b) -> abs(a - b)^p, +, x, y, dims=dims).^(1/p)
            @test vminkowski(x, y, p, dims=dims) ≈ d
            @test vtminkowski(x, y, p, dims=dims) ≈ d
        end
        @test vminkowski(x, y, Inf, dims=dims) ≈ mapreduce(abs ∘ -, max, x, y, dims=dims)
        @test vminkowski(x, y, -Inf, dims=dims) ≈ mapreduce(abs ∘ -, min, x, y, dims=dims)
        @test vtminkowski(x, y, Inf, dims=dims) ≈ mapreduce(abs ∘ -, max, x, y, dims=dims)
        @test vtminkowski(x, y, -Inf, dims=dims) ≈ mapreduce(abs ∘ -, min, x, y, dims=dims)
    end

    for p ∈ (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        d = mapreduce((a, b) -> abs(a - b)^p, +, x, y)^(1/p)
        for dims ∈ ((1,2,3,4), (1,2,3,4,5))
            @test first(vminkowski(x, y, p, dims=dims)) ≈ d
            @test first(vtminkowski(x, y, p, dims=dims)) ≈ d
        end
    end
end

