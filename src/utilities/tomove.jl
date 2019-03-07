####### TO MOVE
import ProximalOperators: prox, prox!, gradient, gradient!

using RecursiveArrayTools
function prox(f::ProximableFunction, x::ArrayPartition, args...) 
  y, fy = prox(f, x.x, args...)
  return ArrayPartition(y), fy
end

prox!(y::ArrayPartition, f::ProximableFunction, x::ArrayPartition, args...) = 
prox!(y.x, f, x.x, args...)

function gradient(f::ProximableFunction, x::ArrayPartition) 
  gradx, fy = gradient(f, x.x)
  return ArrayPartition(gradx), fy
end

gradient!(y::ArrayPartition, f::ProximableFunction, x::ArrayPartition) = 
gradient!(y.x, f, x.x)
###############
