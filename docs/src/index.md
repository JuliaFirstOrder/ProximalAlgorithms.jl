# ProximalAlgorithms.jl

A Julia package for non-smooth optimization algorithms. [Link to GitHub repository](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).

This package provides algorithms for the minimization of objective functions
that include non-smooth terms, such as constraints or non-differentiable penalties.
Implemented algorithms include:
* (Fast) Proximal gradient methods
* Douglas-Rachford splitting
* Three-term splitting
* Primal-dual splitting algorithms
* Newton-type methods

Check out [this section](@ref problems_algorithms) for an overview of the available algorithms.

## Contributing

Contributions are welcome in the form of [issue notifications](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/issues) or [pull requests](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/pulls). When contributing new algorithms, we highly recommend looking at already implemented ones to get inspiration on how to structure the code.

## Table of contents

```@contents
Pages = [
    "getting_started.md",
    "implemented_algorithms.md",
    "custom_objectives.md",
]
Depth = 2
```
