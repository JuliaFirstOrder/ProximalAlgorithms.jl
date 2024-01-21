using Test

using LinearAlgebra
using ProximalAlgorithms
using ProximalAlgorithms:
    NesterovExtrapolation,
    initialize,
    SimpleNesterovSequence,
    FixedNesterovSequence,
    AdaptiveNesterovSequence

for sequence_type in [SimpleNesterovSequence, FixedNesterovSequence]
    for R in [Float32, Float64]
        @testset "Nesterov accel. ($sequence_type, $R)" begin
            H = R[
                0.63287 0.330934 -0.156908 -0.294776 0.10761
                0.330934 0.673201 0.0459778 0.231011 -0.235265
                -0.156908 0.0459778 0.635812 -0.232261 -0.388775
                -0.294776 0.231011 -0.232261 0.726854 -0.0691783
                0.10761 -0.235265 -0.388775 -0.0691783 0.336262
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
            y = x

            norm_initial_error = norm(x_star - x)

            @inferred initialize(NesterovExtrapolation(sequence_type), x)

            seq = sequence_type{R}()
            @test eltype(seq) == R

            @inferred Iterators.peel(seq)

            for (it, coeff) in Iterators.take(enumerate(seq), 100)
                if it == 1
                    @test iszero(coeff)
                end

                grad_f_y = grad_f(y)
                x_prev = x
                x = y - gamma * grad_f_y
                y = x + coeff * (x - x_prev)

                # test that iterates satisfy Thm 4.4 from Beck, Teboulle (2009)
                @test f(x) - f_star <= 2 / (gamma * (it + 1)^2) * norm_initial_error^2
            end
        end
    end
end

@testset "Nesterov adaptive accel." begin
    R = Float64
    fixed_gamma = R(1.7)

    adaptive_seq = AdaptiveNesterovSequence(R(0))
    fixed_seq = FixedNesterovSequence{R}()
    for el in Iterators.take(fixed_seq, 20)
        @test isapprox(el, ProximalAlgorithms.next!(adaptive_seq, fixed_gamma))
    end

    m = R(1)
    adaptive_seq = AdaptiveNesterovSequence(m)
    for _ = 1:20
        @test isapprox(
            (1 - sqrt(m * fixed_gamma)) / (1 + sqrt(m * fixed_gamma)),
            ProximalAlgorithms.next!(adaptive_seq, fixed_gamma),
        )
    end
end
