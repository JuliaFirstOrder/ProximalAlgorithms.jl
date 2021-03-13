struct Noaccel end

function update!(_::Noaccel, _, _) end

import Base: *

function (*)(L::Noaccel, v)
    w = similar(v)
    mul!(w, L, v)
end

import LinearAlgebra: mul!

mul!(d::T, _::Noaccel, v::T) where {T} = copyto!(d, v)
