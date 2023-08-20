# ProximalAlgorithms.jl

[![Build status](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/workflows/CI/badge.svg)](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/JuliaFirstOrder/ProximalAlgorithms.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaFirstOrderProximalAlgorithms.jl?branch=master)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliafirstorder.github.io/ProximalAlgorithms.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliafirstorder.github.io/ProximalAlgorithms.jl/dev)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A Julia package for non-smooth optimization algorithms.

This package provides algorithms for the minimization of objective functions
that include non-smooth terms, such as constraints or non-differentiable penalties.
Implemented algorithms include:
* (Fast) Proximal gradient methods
* Douglas-Rachford splitting
* Three-term splitting
* Primal-dual splitting algorithms
* Newton-type methods

This package works well in combination with [ProximalOperators](https://github.com/JuliaFirstOrder/ProximalOperators.jl) (>= 0.15),
which contains a wide range of functions that can be used to express cost terms.

## Documentation

[Stable version](https://juliafirstorder.github.io/ProximalAlgorithms.jl/stable) (latest release)

[Development version](https://juliafirstorder.github.io/ProximalAlgorithms.jl/dev) (`master` branch)

## Citing

If you use any of the algorithms from ProximalAlgorithms in your research, you are kindly asked to cite the relevant bibliography.
Please check [this section of the manual](https://juliafirstorder.github.io/ProximalAlgorithms.jl/stable/guide/implemented_algorithms/) for algorithm-specific references.

## Contributing

Contributions are welcome in the form of [issues notification](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/issues) or [pull requests](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/pulls). We recommend looking at already implemented algorithms to get inspiration on how to structure new ones.
