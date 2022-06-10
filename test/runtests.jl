using VectorizedReduction
using Test

@testset "VectorizedReduction.jl" begin
    const tests = [
        "vrspecials.jl"
        "reduce.jl",
        "reducedim.jl",
        "treduce.jl",
        "treducedim.jl"
    ]
    for t in tests
        @testset "Test $t" begin
            include(t)
        end
    end
end
