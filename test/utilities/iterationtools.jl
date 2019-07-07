struct FibonacciIterable{I}
    s0::I
    s1::I
end

@testset "IterationTools" begin

    using Printf
    using ProximalAlgorithms: IterationTools
    using Random

    Random.seed!(0)

    Base.iterate(iter::FibonacciIterable) = iter.s0, (iter.s0, iter.s1)
    Base.iterate(iter::FibonacciIterable, state) = state[2], (state[2], sum(state))

    @testset "Looping" begin
        iter = rand(Float64, 10)
        last = IterationTools.loop(iter)
        @test last == iter[end]
    end

    @testset "Halting" begin
        iter = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]
        truncated = IterationTools.halt(iter, x -> x >= 1000)
        @test Base.IteratorSize(truncated) == Base.SizeUnknown()
        @test eltype(truncated) == eltype(iter)
        last = IterationTools.loop(truncated)
        @test last == 1597
    end

    @testset "Side effects" begin
        disp(s) = @info "$s"

        iter = IterationTools.tee(FibonacciIterable(BigInt(0), BigInt(1)), disp)

        fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

        _, state = @test_logs (:info, "$(fib[1])") iterate(iter)
        for k in 2:10
            _, state = @test_logs (:info, "$(fib[k])") iterate(iter, state)
        end
    end

    @testset "Sampling" begin
        iter = randn(Float64, 147)
        sample = IterationTools.sample(iter, 10)
        @test eltype(sample) == Float64
        @test length(sample) == 15
        k = 0
        for x in sample
            idx = min(147, (k+1)*10)
            @test x == iter[idx]
            k = k+1
        end
    end

    @testset "Timing" begin
        iter = randn(Float64, 10)
        timed = IterationTools.stopwatch(iter)
        @test length(timed) == length(iter)
        @test axes(timed) == axes(iter)
        @test eltype(timed) == (UInt64, eltype(iter))
        k = 0
        for (t, x) in timed
            @test x == iter[k+1]
            @test t >= k * 1e8 * 0.9 # 1e8 ns = 0.1 s, factor 0.9 is for safety
            sleep(0.1)
            k = k+1
        end
    end

end
