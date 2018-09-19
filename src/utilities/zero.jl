import ProximalOperators: gradient, gradient!, prox!, prox

struct Zero end

(f::Zero)(x) = 0.0

function gradient!(y, f::Zero, x)
    blockset!(y, 0.0)
    return 0.0
end

function gradient(f::Zero, x)
    y = blockzeros(blocksize(x))
    return y, 0.0
end

function prox!(y, f::Zero, x, gamma)
    blockcopy!(y, x)
    return 0.0
end

function prox(f::Zero, x, gamma)
    y = blockcopy(x)
    return y, 0.0
end
