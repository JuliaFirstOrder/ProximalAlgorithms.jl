const RealOrComplex = Union{R, Complex{R}} where {R <: Real}
const BlockArray = Union{AbstractArray{C1, N} where N, Tuple{Vararg{AbstractArray{C2, N} where {C2 <: RealOrComplex, N}}}} where C1 <: RealOrComplex

# Operations on block-arrays

blocksize(x::Tuple) = blocksize.(x)
blocksize(x::AbstractArray) = size(x)

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
