using VectorizedReduction
using Test
using Random
using LinearAlgebra
using Statistics

const tests = [
    "vrspecials.jl",
    "reduce.jl",
    "reducedim.jl",
    "treduce.jl",
    "treducedim.jl",
    "vmapreducethen.jl",
    "vnorm.jl",
    "vdistance.jl",
    "vstats.jl",
]

# @testset "VectorizedReduction.jl" begin
for t in tests
    @testset "Test $t" begin
        include(t)
    end
end
# end
