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

## Installation

Install the latest stable release with

```julia
julia> ]
pkg> add ProximalAlgorithms
```

To install the development version instead (`master` branch), do

```julia
julia> ]
pkg> add ProximalAlgorithms#master
```

## Citing

If you use any of the algorithms from ProximalAlgorithms in your research, you are kindly asked to cite the relevant bibliography.
Please check [this section of the manual](https://juliafirstorder.github.io/ProximalAlgorithms.jl/dev/implemented_algorithms/) for algorithm-specific references.

## Contributing

Contributions are welcome in the form of [issue notifications](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/issues) or [pull requests](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/pulls). When contributing new algorithms, we highly recommend looking at already implemented ones to get inspiration on how to structure the code.
