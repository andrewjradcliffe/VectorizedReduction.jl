using VectorizedReduction
using Test
using Random
using LinearAlgebra

const tests = [
    "vrspecials.jl",
    "reduce.jl",
    "reducedim.jl",
    "treduce.jl",
    "treducedim.jl",
    "vmapreducethen.jl",
    "vnorm.jl",
    "vdistance.jl",
]

# @testset "VectorizedReduction.jl" begin
for t in tests
    @testset "Test $t" begin
        include(t)
    end
end
# end
