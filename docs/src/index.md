# ProximalAlgorithms.jl

A Julia package for non-smooth optimization algorithms. [Link to GitHub repository](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).

This package provides algorithms for the minimization of objective functions
that include non-smooth terms, such as constraints or non-differentiable penalties.
Implemented algorithms include:
- (Fast) Proximal gradient methods
- Douglas-Rachford splitting
- Three-term splitting
- Primal-dual splitting algorithms
- Newton-type methods

Check out [this section](@ref problems_algorithms) for an overview of the available algorithms.

Algorithms rely on:
- [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) for automatic differentiation
(but you can easily bring your own gradients)
- the [ProximalCore API](https://github.com/JuliaFirstOrder/ProximalCore.jl) for proximal mappings, projections, etc,
to handle non-differentiable terms
(see for example [ProximalOperators](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
for an extensive collection of functions).

!!! note

    ProximalOperators needs to be >=0.15 in order to work with ProximalAlgorithms >=0.5.
    Make sure to update ProximalOperators, in case you have been using versions <0.15.

## Installation

```julia
julia> ]
pkg> add ProximalAlgorithms
```

## Citing

If you use any of the algorithms from ProximalAlgorithms in your research, you are kindly asked to cite the relevant bibliography.
Please check [this section of the manual](@ref problems_algorithms) for algorithm-specific references.

## Contributing

Contributions are welcome in the form of [issue notifications](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/issues) or [pull requests](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/pulls). When contributing new algorithms, we highly recommend looking at already implemented ones to get inspiration on how to structure the code.

