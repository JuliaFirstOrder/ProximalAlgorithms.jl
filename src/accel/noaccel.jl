struct NoAcceleration end

acceleration_style(::Type{<:NoAcceleration}) = NoAccelerationStyle()

initialize(::NoAcceleration, ::Any) = nothing
