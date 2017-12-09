# ProximalAlgorithms

[![Build Status](https://travis-ci.org/kul-forbes/ProximalAlgorithms.jl.svg?branch=master)](https://travis-ci.org/kul-forbes/ProximalAlgorithms.jl)
[![Coverage Status](https://coveralls.io/repos/kul-forbes/ProximalAlgorithms.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/kul-forbes/ProximalAlgorithms.jl?branch=master)
[![codecov.io](http://codecov.io/github/kul-forbes/ProximalAlgorithms.jl/coverage.svg?branch=master)](http://codecov.io/github/kul-forbes/ProximalAlgorithms.jl?branch=master)

Proximal algorithms (also known as "splitting" algorithms) for nonsmooth optimization in Julia.

**Work in progress**, stay tuned for updates and docs.

### Implemented Algorithms

Algorithm                  | Code                                                    | Model          | Reference
---------------------------|---------------------------------------------------------|----------------|----------------------
Douglas-Rachford splitting | [DouglasRachford.jl](src/algorithms/DouglasRachford.jl) | `f(x) + g(x)`  | [[link]][eckstein_bertsekas_1989]
Forward-backward splitting | [ForwardBackward.jl](src/algorithms/ForwardBackward.jl) | `f(Lx) + g(x)` | [[link]][beck_teboulle_2009]
ZeroFPR                    | [ZeroFPR.jl](src/algorithms/ZeroFPR.jl)                 | `f(Lx) + g(x)` | [[link]][themelis_2016]

### References

1. Eckstein, Bertsekas, *On the Douglas-Rachford Splitting Method and the Proximal Point Algorithm for Maximal Monotone Operators*, Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989). [[link]][eckstein_bertsekas_1989]
1. Beck, Teboulle, *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*, SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183-202 (2009). [[link]][beck_teboulle_2009]
1. Boyd, Parikh, Chu, Peleato, Eckstein, *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*, Foundations and Trends in Machine Learning, vol. 3, no. 1, pp. 1-122 (2011). [[link]][boyd_2011]
1. Parikh, Boyd, *Proximal Algorithms*, Foundations and Trends in Optimization, vol. 1, no. 3, pp. 127-239 (2014). [[link]][parikh_2014]
1. Themelis, Stella, Patrinos, *Forward-backward envelope for the sum of two nonconvex functions: Further properties and nonmonotone line-search algorithms*, arXiv:1606.06256 (2016). [[link]][themelis_2016]

[eckstein_bertsekas_1989]: https://link.springer.com/article/10.1007/BF01581204
[beck_teboulle_2009]: http://epubs.siam.org/doi/abs/10.1137/080716542
[boyd_2011]: http://www.nowpublishers.com/article/Details/MAL-016
[parikh_2014]: http://www.nowpublishers.com/article/Details/OPT-003
[themelis_2016]: https://arxiv.org/abs/1606.06256
