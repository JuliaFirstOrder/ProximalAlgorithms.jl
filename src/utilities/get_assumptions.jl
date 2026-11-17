"""
    get_assumptions(::IterativeAlgorithm{IteratorType})
    get_assumptions(::Type{IteratorType})

Return the assumptions on the algorithm `alg` as a tuple of `AssumptionTerm`s.

The returned list is a list of `AssumptionTerm` objects, each of which can be either a `SimpleTerm`,
an `OperatorTerm` or an `OperatorTermWithInfimalConvolution`.
* `SimpleTerm` is used when there is no assumption on the form of the term, so it is assumed to be
in the form of `f(x)`.
* `OperatorTerm` is used when the term is assumed to be in the form of `f(Lx)`, where `f` is a
function and `L` is an operator.
* `OperatorTermWithInfimalConvolution` is used when the term is assumed to be in the form of
`(h □ l)(L x)`, where `f` and `g` are functions, symbol `□` denotes the infimal convolution, and
`L` is an operator.
"""
get_assumptions(::IterativeAlgorithm{IteratorType}) where {IteratorType} = get_assumptions(IteratorType)

const AssumptionItem{T} = Pair{Symbol,T}
abstract type AssumptionTerm end
struct AssumptionGroup{T}
    terms::T
    function AssumptionGroup(terms...)
        @assert all(t -> t isa AssumptionTerm, terms) "All elements of the tuple must be of type AssumptionTerm."
        new{typeof(terms)}(terms)
    end
end

struct LeastSquaresTerm{T} <: AssumptionTerm
    operator::AssumptionItem{T}
    b::Symbol
end

struct SquaredL2Term <: AssumptionTerm
    λ::Symbol
end

struct SimpleTerm{T} <: AssumptionTerm
    func::AssumptionItem{T}
end

struct RepeatedSimpleTerm{T} <: AssumptionTerm
    func::AssumptionItem{T}
end

struct OperatorTerm{T1,T2} <: AssumptionTerm
    func::AssumptionItem{T1}
    operator::AssumptionItem{T2}
end

struct RepeatedOperatorTerm{T1,T2} <: AssumptionTerm
    func::AssumptionItem{T1}
    operator::AssumptionItem{T2}
end

struct OperatorTermWithInfimalConvolution{T1,T2,T3} <: AssumptionTerm
    func₁::AssumptionItem{T1}
    func₂::AssumptionItem{T2}
    operator::AssumptionItem{T3}
end

_show_term(io::IO, t::LeastSquaresTerm) = print(io, "ls(", t.operator.first, "x - ", t.b, ")")
_show_term(io::IO, t::SquaredL2Term) = print(io, t.λ, " * ‖x‖²")
_show_term(io::IO, t::SimpleTerm)  = print(io, t.func.first, "(x)")
_show_term(io::IO, t::RepeatedSimpleTerm) = print(io, t.func.first, "ᵢ", "(x)")
_show_term(io::IO, t::OperatorTerm) = print(io, t.func.first, "(", t.operator.first, "x)")
_show_term(io::IO, t::RepeatedOperatorTerm) = print(io, t.func.first, "ᵢ", "(", t.operator.first, "ᵢ", "x)")
_show_term(io::IO, t::OperatorTermWithInfimalConvolution) = print(io, "(", t.func₁.first, " □ ", t.func₂.first, ")(", t.operator.first, "x)")

_show_properties(io::IO, item::AssumptionItem{T}) where {T} = join(io, item.second, ", ", ", and ")
_show_properties(io::IO, t::LeastSquaresTerm, ::Bool) = begin
    print(io, t.operator.first, " ")
    _show_properties(io, t.operator)
    print(io, " and ", t.b, " is array")
end
_show_properties(io::IO, t::SquaredL2Term, ::Bool) = print(io, t.λ, " is scalar or array")
_show_properties(io::IO, t::SimpleTerm, ::Bool) = begin
    print(io, t.func.first, " ")
    _show_properties(io, t.func)
end
_show_properties(io::IO, t::RepeatedSimpleTerm, ::Bool) = begin
    print(io, t.func.first, "ᵢ", " ")
    _show_properties(io, t.func)
end
_show_properties(io::IO, t::OperatorTerm, newline::Bool) = begin
    print(io, t.func.first, " ")
    _show_properties(io, t.func)
    print(io, newline ? "\n - " : "; and ")
    print(io, t.operator.first, " ")
    if length(t.operator.second) > 0
        _show_properties(io, t.operator)
    end
end
_show_properties(io::IO, t::RepeatedOperatorTerm, newline::Bool) = begin
    print(io, t.func.first, "ᵢ", " ")
    _show_properties(io, t.func)
    print(io, newline ? "\n - " : "; and ")
    print(io, t.operator.first, "ᵢ", " ")
    if length(t.operator.second) > 0
        _show_properties(io, t.operator)
    end
end
_show_properties(io::IO, t::OperatorTermWithInfimalConvolution, newline::Bool) = begin
    if length(t.func₁.second) > 0
        print(io, t.func₁.first, " ")
        _show_properties(io, t.func₁)
        if length(t.func₂.second) > 0 && length(t.operator.second) > 0
            print(io, newline ? "\n - " : "; ")
        elseif length(t.func₂.second) > 0 || length(t.operator.second) > 0
            print(io, newline ? "\n - " : "; and ")
        end
    end
    if length(t.func₂.second) > 0
        print(io, t.func₂.first, " ")
        _show_properties(io, t.func₂)
        if length(t.operator.second) > 0
            print(io, newline ? "\n - " : "; and ")
        end
    end
    if length(t.operator.second) > 0
        print(io, t.operator.first, " ")
        _show_properties(io, t.operator)
    end
end

function show(io::IO, t::AssumptionTerm)
    _show_term(io, t)
    print(io, " where ")
    _show_properties(io, t, false)
end

function show(io::IO, t::AssumptionGroup)
    N = length(t.terms)
    for i in 1:N
        _show_term(io, t.terms[i])
        if i < N
            print(io, " + ")
        end
    end
    print(io, " where ")
    for i in 1:N
        _show_properties(io, t.terms[i], false)
        if i < N - 1
            print(io, "; ")
        elseif i < N
            print(io, "; and ")
        end
    end
end

function show(io::IO, ::MIME"text/plain", t::AssumptionGroup)
    N = length(t.terms)
    for i in 1:N
        _show_term(io, t.terms[i])
        if i < N
            print(io, " + ")
        end
    end
    print(io, " where\n - ")
    for i in 1:N
        _show_properties(io, t.terms[i], true)
        if i < N
            print(io, "\n - ")
        end
    end
end

function Base.iterate(t::AssumptionGroup, state=1)
    state > length(t.terms) && return nothing
    return (t.terms[state], state + 1)
end

function Base.length(t::AssumptionGroup)
    return length(t.terms)
end
