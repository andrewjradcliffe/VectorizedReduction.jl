#
# Date created: 2022-09-14
# Author: aradclif
#
#
############################################################################################

# Statistical things
function vmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> c * z, x, y, dims=dims)
end
function vtmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vtmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> c * z, x, y, dims=dims)
end
function vrmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> √(c * z), x, y, dims=dims)
end
function vtrmse(x, y; dims=:)
    c = 1 / _denom(x, dims)
    vtmapreducethen((xᵢ, yᵢ) -> abs2(xᵢ - yᵢ) , +, z -> √(c * z), x, y, dims=dims)
end

function vmean(f, A; dims=:)
    c = 1 / _denom(A, dims)
    vmapreducethen(f, +, x -> c * x, A, dims=dims)
end
vmean(A; dims=:) = vmean(identity, A, dims=dims)

function vtmean(f, A; dims=:)
    c = 1 / _denom(A, dims)
    vtmapreducethen(f, +, x -> c * x, A, dims=dims)
end
vtmean(A; dims=:) = vtmean(identity, A, dims=dims)

# Naturally, faster than the overflow/underflow-safe logsumexp, but if one can tolerate it...
vlse(A; dims=:) = vmapreducethen(exp, +, log, A, dims=dims)
vtlse(A; dims=:) = vtmapreducethen(exp, +, log, A, dims=dims)
