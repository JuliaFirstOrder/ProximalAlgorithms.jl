####### TO MOVE
using RecursiveArrayTools
function prox(f::ProximableFunction, x::ArrayPartition, args...) 
  y, fy = prox(f, x.x, args...)
  return ArrayPartition(y), fy
end

prox!(y::ArrayPartition, f::ProximableFunction, x::ArrayPartition, args...) = 
prox!(y.x, f, x.x, args...)
###############
