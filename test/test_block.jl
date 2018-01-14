TOL_EQ = 1e-14

x1 = [1.0, 2.0, 3.0, 4.0];
x2 = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0];
x = (x1, x2)

@test typeof(x1) <: BlockArray
@test typeof(x2) <: BlockArray
@test typeof(x) <: BlockArray

@test blocksize(x) == ((4,),(2, 4))
@test blocklength(x) == 12
@test blockvecnorm(x) ≈ 15.297058540778355
@test blockmaxabs(x) == 8

y1 = [4.0, 3.0, 2.0, 1.0];
y2 = [4.0 3.0 2.0 1.0; 8.0 7.0 6.0 5.0];
y = (y1, y2)

@test typeof(y1) <: BlockArray
@test typeof(y2) <: BlockArray
@test typeof(y) <: BlockArray

@test blockvecdot(x, y) == dot(x1, y1) + vecdot(x2, y2)

z = blocksimilar(x)

z .= x .+ y

@test norm(z[1] - x1 - y1) <= TOL_EQ
@test vecnorm(z[2] - x2 - y2) <= TOL_EQ

z .= x .+ 0.1 .* (y .- x)

@test norm(z[1] - x1 - 0.1*(y1 - x1)) <= TOL_EQ
@test vecnorm(z[2] - x2 - 0.1*(y2 - x2)) <= TOL_EQ

for it = 1:100
    z .= y .- 0.2 .* (y .+ x)
end

@test norm(z[1] - y1 + 0.2*(x1 + y1)) <= TOL_EQ
@test vecnorm(z[2] - y2 + 0.2*(x2 + y2)) <= TOL_EQ
