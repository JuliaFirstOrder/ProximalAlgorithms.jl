module IterationTools

export halt, tee, sample, stopwatch, loop

using Base.Iterators

# Halting

struct HaltingIterable{I, F}
    iter::I
    fun::F
end

function Base.iterate(iter::HaltingIterable)
    next = iterate(iter.iter)
    return dispatch(iter, next)
end

function Base.iterate(iter::HaltingIterable, (instruction, state))
    if instruction == :halt return nothing end
    next = iterate(iter.iter, state)
    return dispatch(iter, next)
end

function dispatch(iter::HaltingIterable, next)
    if next === nothing return nothing end
    return next[1], (iter.fun(next[1]) ? :halt : :continue, next[2])
end

halt(iter::I, fun::F) where {I, F} = HaltingIterable{I, F}(iter, fun)

# Side effects

struct TeeIterable{I, F}
    iter::I
    fun::F
end

function Base.iterate(iter::TeeIterable, args...)
    next = iterate(iter.iter, args...)
    if next !== nothing iter.fun(next[1]) end
    return next
end

tee(iter::I, fun::F) where {I, F} = TeeIterable{I, F}(iter, fun)

# Sampling

struct SamplingIterable{I}
    iter::I
    period::UInt
end

function Base.iterate(iter::SamplingIterable, state=iter.iter)
    current = iterate(state)
    if current === nothing return nothing end
    for i = 1:iter.period-1
        next = iterate(state, current[2])
        if next === nothing return current[1], rest(state, current[2]) end
        current = next
    end
    return current[1], rest(state, current[2])
end

sample(iter::I, period) where I = SamplingIterable{I}(iter, period)

# Timing

struct StopwatchIterable{I}
    iter::I
end

function Base.iterate(iter::StopwatchIterable)
    t0 = time_ns()
    next = iterate(iter.iter)
    return dispatch(iter, t0, next)
end

function Base.iterate(iter::StopwatchIterable, (t0, state))
    next = iterate(iter.iter, state)
    return dispatch(iter, t0, next)
end

function dispatch(iter::StopwatchIterable, t0, next)
    if next === nothing return nothing end
    return (time_ns()-t0, next[1]), (t0, next[2])
end

stopwatch(iter::I) where I = StopwatchIterable{I}(iter)

# Looping

function loop(iter)
    x = nothing
    for y in iter x = y end
    return x
end

end # module
