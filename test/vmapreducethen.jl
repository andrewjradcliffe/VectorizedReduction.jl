# Assorted tests; TODO: systematic
Random.seed!(0x9f164f673dbb3f65)
@testset "Simple mapreduce with post-operator" begin
    xx = rand(ntuple(_ -> 3, 4)...);
    R = vmapreducethen(abs2, +, √, xx, dims=(2,4))
    R2 = .√mapreduce(abs2, +, xx, dims=(2,4))
    @test R ≈ R2
end

@testset "mapreducethen, varargs" begin
    x = rand(5,5,5,5);
    y = rand(5,5,5,5);
    z = rand(5,5,5,5);
    @test vmapreducethen(+, +, abs2, x, y, z) ≈ abs2(mapreduce(+, +, x, y, z))
    for f ∈ (+, *, muladd)
        for op ∈ (+, *, max, min)
            for g ∈ (abs, abs2, sqrt)
                for dims ∈ (1, 2, 3, 4, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), (1,2,3,4,5))
                    @test vmapreducethen(f, op, g, x, y, z, dims=dims) ≈ g.(mapreduce(f, op, x, y, z, dims=dims))
                end
            end
        end
    end
end
