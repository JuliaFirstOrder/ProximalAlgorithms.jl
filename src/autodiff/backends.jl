struct YotaFunction{F}
    f::F
end

(f::YotaFunction)(x) = f.f(x)

struct ZygoteFunction{F}
    f::F
end

(f::ZygoteFunction)(x) = f.f(x)
