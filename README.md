# ProximalAlgorithms.jl

[![Build status](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/workflows/CI/badge.svg)](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/JuliaFirstOrder/ProximalAlgorithms.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaFirstOrderProximalAlgorithms.jl?branch=master)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliafirstorder.github.io/ProximalAlgorithms.jl/dev)

A Julia package for non-smooth optimization algorithms.

This package provides algorithms for the minimization of objective functions
that include non-smooth terms, such as constraints or non-differentiable penalties.
Implemented algorithms include:
* (Fast) Proximal gradient methods
* Douglas-Rachford splitting
* Three-term splitting
* Primal-dual splitting algorithms
* Newton-type methods

## Documentation

[Development version (`master` branch)](https://juliafirstorder.github.io/ProximalAlgorithms.jl/dev)

## Contributing

Contributions are welcome in the form of [issues notification](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/issues) or [pull requests](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/pulls). We recommend looking at already implemented algorithms to get inspiration on how to structure new ones.

## Related packages

This package can be used in combination with [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) (providing first-order primitives, i.e. gradient and proximal mapping, for numerous cost functions) and [AbstractOperators.jl](https://github.com/kul-forbes/AbstractOperators.jl) (providing several linear and nonlinear operators) to formulate and solve a wide spectrum of nonsmooth optimization problems.
[StructuredOptimization.jl](https://github.com/JuliaFirstOrder/StructuredOptimization.jl) provides a higher-level interface to formulate and solve problems using (some of) the algorithms here included.
