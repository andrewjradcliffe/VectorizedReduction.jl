Random.seed!(0xe89edc3a6fb53d11)
@testset "1-dimensional norms" begin
    x = rand(1000)
    for p ∈ (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, Inf, -Inf)
        d = norm(x, p)
        @test vnorm(x, p) ≈ d
        @test vtnorm(x, p) ≈ d
    end
end

@testset "multi-dimensional norms" begin
    x = rand(10,10,10,10);
    for dims ∈ (1, 2, 3, 4, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4))
        x′ = eachslice(x, dims=Tuple(setdiff(1:4, dims)))
        for p ∈ (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, Inf, -Inf)
            d = norm.(x′, p)
            @test dropdims(vnorm(x, p, dims=dims), dims=dims) ≈ d
            @test dropdims(vtnorm(x, p, dims=dims), dims=dims) ≈ d
        end
    end
    for p ∈ (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, Inf, -Inf)
        d = norm(x, p)
        for dims ∈ ((1,2,3,4), (1,2,3,4,5))
            @test first(vnorm(x, p, dims=dims)) ≈ d
            @test first(vtnorm(x, p, dims=dims)) ≈ d
        end
    end
end
