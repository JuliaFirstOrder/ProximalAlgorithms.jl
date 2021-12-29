abstract type AccelerationStyle end

struct UnknownStyle <: AccelerationStyle end

struct NoAccelerationStyle <: AccelerationStyle end

struct QuasiNewtonStyle <: AccelerationStyle end

struct NesterovStyle <: AccelerationStyle end

acceleration_style(::Type{<:Any}) = UnknownStyle()
