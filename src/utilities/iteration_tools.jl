module IterationTools

export halt, tee, sample, stopwatch, loop

using Base.Iterators

# Halting

struct HaltingIterable{I,F}
    iter::I
    fun::F
end

Base.IteratorSize(::Type{HaltingIterable{I,F}}) where {I,F} = Base.IteratorSize(I)
Base.IteratorEltype(::Type{HaltingIterable{I,F}}) where {I,F} = Base.IteratorEltype(I)

Base.length(iter::HaltingIterable{I,F}) where {I,F} = length(iter.iter)
Base.eltype(iter::HaltingIterable{I,F}) where {I,F} = eltype(iter.iter)

function Base.iterate(iter::HaltingIterable)
    next = iterate(iter.iter)
    return dispatch(iter, next)
end

function Base.iterate(iter::HaltingIterable, (instruction, state))
    if instruction == :halt
        return nothing
    end
    next = iterate(iter.iter, state)
    return dispatch(iter, next)
end

function dispatch(iter::HaltingIterable, next)
    if next === nothing
        return nothing
    end
    return next[1], (iter.fun(next[1]) ? :halt : :continue, next[2])
end

halt(iter::I, fun::F) where {I,F} = HaltingIterable{I,F}(iter, fun)

# Side effects

struct TeeIterable{I,F}
    iter::I
    fun::F
end

Base.IteratorSize(::Type{TeeIterable{I,F}}) where {I,F} = Base.IteratorSize(I)
Base.IteratorEltype(::Type{TeeIterable{I,F}}) where {I,F} = Base.IteratorEltype(I)

Base.length(iter::TeeIterable{I,F}) where {I,F} = length(iter.iter)
Base.axes(iter::TeeIterable{I,F}) where {I,F} = axes(iter.iter)
Base.eltype(iter::TeeIterable{I,F}) where {I,F} = eltype(iter.iter)

function Base.iterate(iter::TeeIterable, args...)
    next = iterate(iter.iter, args...)
    if next !== nothing
        iter.fun(next[1])
    end
    return next
end

tee(iter::I, fun::F) where {I,F} = TeeIterable{I,F}(iter, fun)

# Sampling

struct SamplingIterable{I}
    iter::I
    period::UInt
end

Base.IteratorSize(::Type{SamplingIterable{I}}) where {I} = Base.IteratorSize(I)
Base.IteratorEltype(::Type{SamplingIterable{I}}) where {I} = Base.IteratorEltype(I)

function Base.length(iter::SamplingIterable{I}) where {I}
    remainder = length(iter.iter) % iter.period
    quotient = length(iter.iter) ÷ iter.period
    return remainder == 0 ? quotient : quotient + 1
end

Base.size(iter::SamplingIterable{I}) where {I} = (length(iter),)
Base.eltype(iter::SamplingIterable{I}) where {I} = eltype(iter.iter)

function Base.iterate(iter::SamplingIterable, state = iter.iter)
    current = iterate(state)
    if current === nothing
        return nothing
    end
    for i = 1:iter.period-1
        next = iterate(state, current[2])
        if next === nothing
            return current[1], rest(state, current[2])
        end
        current = next
    end
    return current[1], rest(state, current[2])
end

sample(iter::I, period) where {I} = SamplingIterable{I}(iter, period)

# Timing

struct StopwatchIterable{I}
    iter::I
end

Base.IteratorSize(::Type{StopwatchIterable{I}}) where {I} = Base.IteratorSize(I)
Base.IteratorEltype(::Type{StopwatchIterable{I}}) where {I} = Base.IteratorEltype(I)

Base.length(iter::StopwatchIterable{I}) where {I} = length(iter.iter)
Base.axes(iter::StopwatchIterable{I}) where {I} = axes(iter.iter)
Base.eltype(iter::StopwatchIterable{I}) where {I} = (UInt64, eltype(iter.iter))

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
    if next === nothing
        return nothing
    end
    return (time_ns() - t0, next[1]), (t0, next[2])
end

stopwatch(iter::I) where {I} = StopwatchIterable{I}(iter)

# Looping

function loop(iter)
    output, state = iterate(iter)
    res = iterate(iter, state)
    while res !== nothing
        output, state = res
        res = iterate(iter, state)
    end
    return output
end

end # module
