using Test

using LinearAlgebra
using ProximalAlgorithms: NesterovExtrapolation, initialize, SimpleNesterovSequence, NesterovSequence

for sequence_type in [SimpleNesterovSequence, NesterovSequence]
    for R in [Float32, Float64]
        @testset "Nesterov accel. ($sequence_type, $R)" begin
            H = R[
                0.63287    0.330934   -0.156908   -0.294776    0.10761;
                0.330934   0.673201    0.0459778   0.231011   -0.235265;
               -0.156908   0.0459778   0.635812   -0.232261   -0.388775;
               -0.294776   0.231011   -0.232261    0.726854   -0.0691783;
                0.10761   -0.235265   -0.388775   -0.0691783   0.336262;
            ]
            l = R[1.0, 2.0, 3.0, 4.0, 5.0]

            n = length(l)

            f(x) = dot(x, H * x) / 2 + dot(x, l)
            grad_f(x) = H * x + l

            x_star = -H \ l
            f_star = f(x_star)

            Lip = opnorm(H)
            gamma = 1 / Lip
            x = zeros(R, n)
            x_prev = x

            norm_initial_error = norm(x_star - x)

            @inferred initialize(NesterovExtrapolation(sequence_type), x)

            seq = sequence_type{R}()

            @inferred Iterators.peel(seq)

            for (it, coeff) in Iterators.take(enumerate(seq), 100)
                if it <= 2
                    @test iszero(coeff)
                end

                y = x + coeff * (x - x_prev)
                grad_f_y = grad_f(y)
                x_prev = x
                x = y - gamma * grad_f_y

                # test that iterates satisfy Thm 4.4 from Beck, Teboulle (2009)
                @test f(x) - f_star <= 2 / (gamma * (it + 1)^2) * norm_initial_error^2
            end
        end
    end
end
