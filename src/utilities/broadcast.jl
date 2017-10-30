import Base: broadcast!

function broadcast!(f::Any, dest::Tuple, op1::Tuple, op2::Tuple)
   for k = eachindex(dest)
       broadcast!(f, dest[k], op1[k], op2[k])
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
