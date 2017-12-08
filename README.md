# ProximalAlgorithms

[![Build Status](https://travis-ci.org/kul-forbes/ProximalAlgorithms.jl.svg?branch=master)](https://travis-ci.org/kul-forbes/ProximalAlgorithms.jl)
[![Coverage Status](https://coveralls.io/repos/kul-forbes/ProximalAlgorithms.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/kul-forbes/ProximalAlgorithms.jl?branch=master)
[![codecov.io](http://codecov.io/github/kul-forbes/ProximalAlgorithms.jl/coverage.svg?branch=master)](http://codecov.io/github/kul-forbes/ProximalAlgorithms.jl?branch=master)

Proximal algorithms (also known as "splitting" algorithms) for nonsmooth optimization in Julia.

**Work in progress**, stay tuned for updates and docs.

### Implemented Algorithms

Algorithm                  | Code                                                    | Model          | Reference
---------------------------|---------------------------------------------------------|----------------|----------------------
Forward-backward splitting | [ForwardBackward.jl](src/algorithms/ForwardBackward.jl) | `f(Lx) + g(x)` | [[link]][1]
ZeroFPR                    | [ZeroFPR.jl](src/algorithms/ZeroFPR.jl)                 | `f(Lx) + g(x)` | [[link]][2]
Douglas-Rachford splitting | [DouglasRachford.jl](src/algorithms/DouglasRachford.jl) | `f(x) + g(x)`  | [[link]][3]

### References

1. Beck, Teboulle, *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*, SIAM Journal on Imaging Sciences vol. 2, no. 1, pp. 183-202 (2009). [[link]][1]
2. Eckstein, Bertsekas, *On the Douglas-Rachford Splitting Method and the Proximal Point Algorithm for Maximal Monotone Operators*, Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989). [[link]][2]
3. Themelis, Stella, Patrinos, *Forward-backward envelope for the sum of two nonconvex functions: Further properties and nonmonotone line-search algorithms*, arXiv:1606.06256 (2017). [[link]][3]

[1]: http://epubs.siam.org/doi/abs/10.1137/080716542
[2]: https://link.springer.com/article/10.1007/BF01581204
[3]: https://arxiv.org/abs/1606.06256
