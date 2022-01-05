# ProximalAlgorithms.jl

[![Build status](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/workflows/CI/badge.svg)](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/JuliaFirstOrder/ProximalAlgorithms.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaFirstOrderProximalAlgorithms.jl?branch=master)

Proximal algorithms (also known as "splitting" algorithms or methods) for nonsmooth optimization in Julia.

This package can be used in combination with [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) (providing first-order primitives, i.e. gradient and proximal mapping, for numerous cost functions) and [AbstractOperators.jl](https://github.com/kul-forbes/AbstractOperators.jl) (providing several linear and nonlinear operators) to formulate and solve a wide spectrum of nonsmooth optimization problems.

[StructuredOptimization.jl](https://github.com/JuliaFirstOrder/StructuredOptimization.jl) provides a higher-level interface to formulate and solve problems using (some of) the algorithms here included.

### Quick start

To install the package, simply issue the following command in the Julia REPL:

```julia
] add ProximalAlgorithms
```

Check out [these test scripts](test/problems) for examples on how to apply
the provided algorithms to problems.

### Implemented Algorithms

Algorithm                             | Function      
--------------------------------------|---------------
Douglas-Rachford splitting[^eckstein_1989] | [`DouglasRachford`](src/algorithms/douglas_rachford.jl)
Forward-backward splitting (i.e. proximal gradient)[^lions_mercier_1979] | [`ForwardBackward`](src/algorithms/forward_backward.jl)
Fast forward-backward splitting[^tseng_2008][^beck_2009] | [`FastForwardBackward`](src/algorithms/forward_backward.jl)
Vũ-Condat primal-dual algorithm[^chambolle_2011][^vu_2013][^condat_2013] | [`VuCondat`](src/algorithms/primal_dual.jl)
Davis-Yin splitting[^davis_2017] | [`DavisYin`](src/algorithms/davis_yin.jl)
Asymmetric forward-backward-adjoint splitting[^latafat_2017] | [`AFBA`](src/algorithms/primal_dual.jl)
PANOC (L-BFGS)[^stella_2017] | [`PANOC`](src/algorithms/panoc.jl)
PANOC+ (L-BFGS)[^demarchi_2021] | [`PANOCplus`](src/algorithms/panocplus.jl)
ZeroFPR (L-BFGS)[^themelis_2018] | [`ZeroFPR`](src/algorithms/zerofpr.jl)
Douglas-Rachford line-search (L-BFGS)[^themelis_2020] | [`DRLS`](src/algorithms/drls.jl)

### Contributing

Contributions are welcome in the form of [issues notification](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/issues) or [pull requests](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/pulls). We recommend looking at already implemented algorithms to get inspiration on how to structure new ones.

[^lions_mercier_1979]: Lions, Mercier, “Splitting algorithms for the sum of two nonlinear operators,” SIAM Journal on Numerical Analysis, vol. 16, pp. 964–979 (1979). [link](https://epubs.siam.org/doi/abs/10.1137/0716071)

[^eckstein_1989]: Eckstein, Bertsekas, *On the Douglas-Rachford Splitting Method and the Proximal Point Algorithm for Maximal Monotone Operators*, Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989). [link](https://link.springer.com/article/10.1007/BF01581204)

[^tseng_2008]: Tseng, *On Accelerated Proximal Gradient Methods for Convex-Concave Optimization* (2008). [link](http://www.mit.edu/~dimitrib/PTseng/papers/apgm.pdf)

[^beck_2009]: Beck, Teboulle, *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*, SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183-202 (2009). [link](http://epubs.siam.org/doi/abs/10.1137/080716542)

[^chambolle_2011]: Chambolle, Pock, *A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging*, Journal of Mathematical Imaging and Vision, vol. 40, no. 1, pp. 120-145 (2011). [link](https://link.springer.com/article/10.1007/s10851-010-0251-1)

[^boyd_2011]: Boyd, Parikh, Chu, Peleato, Eckstein, *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*, Foundations and Trends in Machine Learning, vol. 3, no. 1, pp. 1-122 (2011). [link](http://www.nowpublishers.com/article/Details/MAL-016)

[^vu_2013]: Vũ, *A splitting algorithm for dual monotone inclusions involving cocoercive operators*, Advances in Computational Mathematics, vol. 38, no. 3, pp. 667-681 (2013). [link](https://link.springer.com/article/10.1007/s10444-011-9254-8)

[^condat_2013]: Condat, *A primal–dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms*, Journal of Optimization Theory and Applications, vol. 158, no. 2, pp 460-479 (2013). [link](https://link.springer.com/article/10.1007/s10957-012-0245-9)

[^parikh_2014]: Parikh, Boyd, *Proximal Algorithms*, Foundations and Trends in Optimization, vol. 1, no. 3, pp. 127-239 (2014). [link](http://www.nowpublishers.com/article/Details/OPT-003)

[^davis_2017]: Davis, Yin, *A Three-Operator Splitting Scheme and its Optimization Applications*, Set-Valued and Variational Analysis, vol. 25, no. 4, pp. 829–858 (2017). [link](https://link.springer.com/article/10.1007/s11228-017-0421-z)

[^latafat_2017]: Latafat, Patrinos, *Asymmetric forward–backward–adjoint splitting for solving monotone inclusions involving three operators*, Computational Optimization and Applications, vol. 68, no. 1, pp. 57-93 (2017). [link](https://link.springer.com/article/10.1007/s10589-017-9909-6)

[^stella_2017]: Stella, Themelis, Sopasakis, Patrinos, *A simple and efficient algorithm for nonlinear model predictive control*, 56th IEEE Conference on Decision and Control (2017). [link](https://doi.org/10.1109/CDC.2017.8263933)

[^themelis_2018]: Themelis, Stella, Patrinos, *Forward-backward envelope for the sum of two nonconvex functions: Further properties and nonmonotone line-search algorithms*, SIAM Journal on Optimization, vol. 28, no. 3, pp. 2274–2303 (2018). [link](https://epubs.siam.org/doi/10.1137/16M1080240)

[^themelis_2020]: Themelis, Stella, Patrinos, *Douglas-Rachford splitting and ADMM for nonconvex optimization: Accelerated and Newton-type algorithms*, arXiv preprint (2020). [link](https://arxiv.org/abs/2005.10230)

[^demarchi_2021]: De Marchi, Themelis, *Proximal gradient algorithms under local Lipschitz gradient continuity: a convergence and robustness analysis of PANOC*, arXiv preprint (2021). [link](https://arxiv.org/abs/2112.13000)
