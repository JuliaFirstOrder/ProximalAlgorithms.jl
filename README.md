# ProximalAlgorithms.jl

[![Build Status](https://travis-ci.org/kul-forbes/ProximalAlgorithms.jl.svg?branch=master)](https://travis-ci.org/kul-forbes/ProximalAlgorithms.jl)
[![Coverage Status](https://coveralls.io/repos/kul-forbes/ProximalAlgorithms.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/kul-forbes/ProximalAlgorithms.jl?branch=master)
[![codecov.io](http://codecov.io/github/kul-forbes/ProximalAlgorithms.jl/coverage.svg?branch=master)](http://codecov.io/github/kul-forbes/ProximalAlgorithms.jl?branch=master)

Proximal algorithms (also known as "splitting" algorithms or methods) for nonsmooth optimization in Julia.

This package can be used in combination with [ProximalOperators.jl](https://github.com/kul-forbes/ProximalOperators.jl) (providing first-order primitives, i.e. gradient and proximal mapping, for numerous cost functions) and [AbstractOperators.jl](https://github.com/kul-forbes/AbstractOperators.jl) (providing several linear and nonlinear operators) to formulate and solve a wide spectrum of nonsmooth optimization problems.

[StructuredOptimization.jl](https://github.com/kul-forbes/StructuredOptimization.jl) provides a higher-level interface to formulate and solve problems using (some of) the algorithms here included.

### Installation

```julia
julia> Pkg.add("ProximalAlgorithms")
```

### Implemented Algorithms

Algorithm                             | Function      | Reference
--------------------------------------|---------------|-----------
Asymmetric forward-backward-adjoint algorithm | [`AFBA`](src/algorithms/AsymmetricForwardBackwardAdjoint.jl) | [[10]][latafat_2017]
Chambolle-Pock primal dual algorithm  | [`ChambollePock`](src/algorithms/AsymmetricForwardBackwardAdjoint.jl) | [[4]][chambolle_2011]
Douglas-Rachford splitting algorithm  | [`DRS`](src/algorithms/DouglasRachford.jl) | [[1]][eckstein_1989]
Forward-backward splitting (i.e. proximal gradient) algorithm | [`FBS`](src/algorithms/ForwardBackward.jl) | [[2]][tseng_2008], [[3]][beck_2009]
Vũ-Condat primal-dual algorithm       | [`VuCondat`](src/algorithms/AsymmetricForwardBackwardAdjoint.jl) | [[6]][vu_2013], [[7]][condat_2013]
ZeroFPR (L-BFGS)                      | [`ZeroFPR`](src/algorithms/ZeroFPR.jl) | [[9]][themelis_2016]
PANOC (L-BFGS)                        | [`PANOC`](src/algorithms/PANOC.jl) | [[11]][stella_2017]

### Contributing

Contributions are welcome in the form of [issues notification](https://github.com/kul-forbes/ProximalAlgorithms.jl/issues) or [pull requests](https://github.com/kul-forbes/ProximalAlgorithms.jl/pulls). We recommend looking at already implemented algorithms, or following the [template](src/template/Template.jl), to get inspiration on how to structure new ones.

### References

[[1]][eckstein_1989] Eckstein, Bertsekas, *On the Douglas-Rachford Splitting Method and the Proximal Point Algorithm for Maximal Monotone Operators*, Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).

[[2]][tseng_2008] Tseng, *On Accelerated Proximal Gradient Methods for Convex-Concave Optimization* (2008).

[[3]][beck_2009] Beck, Teboulle, *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*, SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183-202 (2009).

[[4]][chambolle_2011] Chambolle, Pock, *A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging*, Journal of Mathematical Imaging and Vision, vol. 40, no. 1, pp. 120-145 (2011).

[[5]][boyd_2011] Boyd, Parikh, Chu, Peleato, Eckstein, *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*, Foundations and Trends in Machine Learning, vol. 3, no. 1, pp. 1-122 (2011).

[[6]][vu_2013] Vũ, *A splitting algorithm for dual monotone inclusions involving cocoercive operators*, Advances in Computational Mathematics, vol. 38, no. 3, pp. 667-681 (2013).

[[7]][condat_2013] Condat, *A primal–dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms*, Journal of Optimization Theory and Applications, vol. 158, no. 2, pp 460-479 (2013).

[[8]][parikh_2014] Parikh, Boyd, *Proximal Algorithms*, Foundations and Trends in Optimization, vol. 1, no. 3, pp. 127-239 (2014).

[[9]][themelis_2016] Themelis, Stella, Patrinos, *Forward-backward envelope for the sum of two nonconvex functions: Further properties and nonmonotone line-search algorithms*, arXiv:1606.06256 (2016).

[[10]][latafat_2017] Latafat, Patrinos, *Asymmetric forward–backward–adjoint splitting for solving monotone inclusions involving three operators*, Computational Optimization and Applications, vol. 68, no. 1, pp. 57-93 (2017).

[[11]][stella_2017] Stella, Themelis, Sopasakis, Patrinos, *A simple and efficient algorithm for nonlinear model predictive control*, 56th IEEE Conference on Decision and Control (2017).

[eckstein_1989]: https://link.springer.com/article/10.1007/BF01581204
[tseng_2008]: http://www.mit.edu/~dimitrib/PTseng/papers/apgm.pdf
[beck_2009]: http://epubs.siam.org/doi/abs/10.1137/080716542
[chambolle_2011]: https://link.springer.com/article/10.1007/s10851-010-0251-1
[boyd_2011]: http://www.nowpublishers.com/article/Details/MAL-016
[parikh_2014]: http://www.nowpublishers.com/article/Details/OPT-003
[themelis_2016]: https://arxiv.org/abs/1606.06256
[latafat_2017]: https://link.springer.com/article/10.1007/s10589-017-9909-6
[stella_2017]: https://doi.org/10.1109/CDC.2017.8263933
[condat_2013]: https://link.springer.com/article/10.1007/s10957-012-0245-9
[vu_2013]: https://link.springer.com/article/10.1007/s10444-011-9254-8
