using VectorizedReduction
using Test

const tests = [
    "vrspecials.jl",
    "reduce.jl",
    "reducedim.jl",
    "treduce.jl",
    "treducedim.jl"
]

# @testset "VectorizedReduction.jl" begin
for t in tests
    @testset "Test $t" begin
        include(t)
    end
end
# end
