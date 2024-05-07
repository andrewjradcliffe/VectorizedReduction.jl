using VectorizedReduction
using Test
using Random
using LinearAlgebra
using Statistics

macro inctests(xs...)
    block = Expr(:block)
    for x in xs
        e = :(@testset "$($x)" begin
                  include("$($x).jl")
              end)
        push!(block.args, e)
    end
    quote
        $block
    end
end

@inctests "vrspecials" "reduce" "reducedim" "treduce" "treducedim"
@inctests "vmapreducethen" "vnorm" "vdistance" "vstats"
