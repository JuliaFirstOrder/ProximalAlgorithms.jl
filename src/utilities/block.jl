# Define block-arrays

const RealOrComplex{R} = Union{R, Complex{R}}
const BlockArray{R} = Union{
	AbstractArray{C, N} where {C <: RealOrComplex{R}, N},
	Tuple{Vararg{AbstractArray{C, N} where {C <: RealOrComplex{R}, N}}}
}

# Operations on block-arrays

blocksize(x::Tuple) = blocksize.(x)
blocksize(x::AbstractArray) = size(x)

blockeltype(x::Tuple) = blockeltype.(x)
blockeltype(x::AbstractArray) = eltype(x)

blocklength(x::Tuple) = sum(blocklength.(x))
blocklength(x::AbstractArray) = length(x)

blockvecnorm(x::Tuple) = sqrt(blockvecdot(x, x))
blockvecnorm{R <: Number}(x::AbstractArray{R}) = vecnorm(x)

blockmaxabs(x::Tuple) = maximum(blockmaxabs.(x))
blockmaxabs{R <: Number}(x::AbstractArray{R}) = maximum(abs, x)

blocksimilar(x::Tuple) = blocksimilar.(x)
blocksimilar(x::AbstractArray) = similar(x)

blockcopy(x::Tuple) = blockcopy.(x)
blockcopy(x::Array) = copy(x)

blockcopy!(y::Tuple, x::Tuple) = blockcopy!.(y, x)
blockcopy!(y::AbstractArray, x::AbstractArray) = copy!(y, x)

blockset!(y::Tuple, x) = blockset!.(y, x)
blockset!(y::AbstractArray, x) = (y .= x)

blockvecdot{T <: Tuple}(x::T, y::T) = sum(blockvecdot.(x,y))
blockvecdot{R <: Number}(x::AbstractArray{R}, y::AbstractArray{R}) = vecdot(x, y)

blockzeros(t::Tuple, s::Tuple) = blockzeros.(t, s)
blockzeros(t::Type, n::NTuple{N, Integer} where {N}) = zeros(t, n)
blockzeros(t::Tuple) = blockzeros.(t)
blockzeros(n::NTuple{N, Integer} where {N}) = zeros(n)
blockzeros(n::Integer) = zeros(n)
blockzeros(a::AbstractArray) = zeros(a)

blockaxpy!(z::Tuple, x, alpha::Real, y::Tuple) = blockaxpy!.(z, x, alpha, y)
blockaxpy!(z::AbstractArray, x, alpha::Real, y::AbstractArray) = (z .= x .+ alpha.*y)

# Define broadcast

import Base: broadcast!

function broadcast!(f::Any, dest::Tuple, op1::Tuple)
   for k = eachindex(dest)
       broadcast!(f, dest[k], op1[k])
   end
   return dest
end

function broadcast!(f::Any, dest::Tuple, op1::Tuple, op2::Tuple)
   for k = eachindex(dest)
       broadcast!(f, dest[k], op1[k], op2[k])
   end
   return dest
end

function broadcast!(f::Any, dest::Tuple, coef::Number, op2::Tuple)
   for k = eachindex(dest)
       broadcast!(f, dest[k], coef, op2[k])
   end
   return dest
end

function broadcast!(f::Any, dest::Tuple, op1::Tuple, coef::Number, op2::Tuple)
   for k = eachindex(dest)
       broadcast!(f, dest[k], op1[k], coef, op2[k])
   end
   return dest
end

function broadcast!(f::Any, dest::Tuple, op1::Tuple, coef::Number, op2::Tuple, op3::Tuple)
   for k = eachindex(dest)
       broadcast!(f, dest[k], op1[k], coef, op2[k], op3[k])
   end
   return dest
end
