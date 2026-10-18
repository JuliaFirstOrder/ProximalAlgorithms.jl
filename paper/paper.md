---
title: 'ProximalAlgorithms.jl for nonsmooth optimization in Julia'
tags:
  - Julia
  - nonsmooth optimization
  - nonconvex optimization
  - convex optimization
  - proximal algorithms
  - splitting methods
authors:
  - name: Lorenzo Stella
    affiliation: 1
  - name: Niccoló Antonello
    orcid: 0000-0002-0803-5385
    affiliation: 2
  - name: Puya Latafat
    orcid: 0000-0002-7969-8565
    affiliation: 3
  - name: Alberto De Marchi
    orcid: 0000-0002-3545-6898
    corresponding: true
    affiliation: 4
affiliations:
 - name: AWS AI Labs, Berlin, Germany (work done prior to joining Amazon)
   index: 1
 - name: EssilorLuxottica, Milan, Italy (work done prior to joining EssilorLuxottica)
   index: 2
 - name: IMT School for Advanced Studies Lucca, Italy
   index: 3
 - name: University of the Bundeswehr Munich, Germany
   index: 4
date: xx xxxxxxx 202x
bibliography: paper.bib
header-includes: |
  \usepackage{booktabs}
---

# Summary

[ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) is a Julia package that implements a broad collection of first-order splitting methods, commonly known as *proximal algorithms*, for solving structured nonsmooth optimization problems of the form
\begin{equation}\label{eq:generic}
    \underset{x \in \mathbb{R}^n}{\text{minimize}} \quad \sum_{i=1}^{N} f_i(x),
\end{equation}
where $N$ depends on the algorithm of choice, and each term $f_i$ may be smooth or nonsmooth, convex or, for some algorithms, nonconvex.
Terms can also be composed with a linear mapping, as in $h(Lx)$, which arises frequently in imaging, signal processing, and control applications.
Rather than assuming access to explicit derivatives or closed-form solutions, proximal algorithms rely only on first-order information about each term: its gradient $\nabla f_i$, when it is differentiable, or its proximal mapping
\begin{equation}\label{eq:prox}
    \operatorname{prox}_{\gamma f_i}(v) = \underset{x \in \mathbb{R}^n}{\arg\min} \left\{ f_i(x) + \frac{1}{2\gamma}\|x - v\|^2 \right\},
\end{equation}
when it is not.
This makes the methods well suited to large-scale problems in which nonsmooth terms (indicator functions of simple sets, sparsity-inducing penalties, norms, etc.) admit an inexpensive proximal mapping, even though the objective as a whole is not directly minimizable in closed form.

`ProximalAlgorithms.jl` organizes its solvers according to the structure of the problem they address:

- **Two-term splitting** ($f + g$): (Fast) proximal gradient / forward-backward splitting [@lions1979; @beck2009], Douglas-Rachford splitting [@eckstein1992], and the line-search-based methods ZeroFPR [@themelis2018], PANOC [@stella2017], PANOC+ [@demarchi2022], and Douglas-Rachford line search (DRLS) [@themelis2022].
- **Three-term splitting** ($f + g + h$): the Davis-Yin splitting scheme [@davis2017].
- **Primal-dual splitting** ($f + g + h \circ L$): Chambolle-Pock [@chambolle2011], Vũ-Condat [@vu2013; @condat2013], and the asymmetric forward-backward-adjoint (AFBA) method [@latafat2017].

Adaptive schemes for convex problems are being introduced as well, particularly AdaPGM and AdaPDM [@latafat2025].

# Statement of need

Nonsmooth structured optimization problems arise pervasively in statistics, machine learning, signal and image processing, and optimal control, wherever nonsmooth regularizers or constraints are combined with a differentiable loss or data-fidelity term.
Because no single algorithm applies uniformly well to every instance of this problem class, practitioners benefit from a library that offers many interchangeable methods behind a common interface, rather than a single monolithic solver.
Moreover, methods differ in the number and structure of terms they support, in the convexity and smoothness assumptions they require, and in their empirical convergence behavior.

# Software design

All solvers in `ProximalAlgorithms.jl` are implemented as Julia iterators, so that users can either call a solver directly to obtain a solution, or loop over the underlying iterator to gain fine-grained control over logging, stopping criteria, or warm-starting.
Gradients are obtained, by default, through automatic differentiation via [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl), which gives access to essentially every Julia automatic-differentiation backend, while proximal mappings follow the lightweight [ProximalCore.jl](https://github.com/JuliaFirstOrder/ProximalCore.jl) interface implemented, among others, by [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl).
Users can just as easily supply hand-written gradients or proximal mappings for custom objective terms.

`ProximalAlgorithms.jl` deliberately does not provide a modeling language that automatically decomposes an arbitrary objective into terms and picks a matching algorithm, in the way that, e.g., disciplined convex programming frameworks do for problems expressible in their grammar.
Instead, the user formulates the problem directly in terms of the objective terms ($f$, $g$, $h$, and, where relevant, a linear operator $L$) that a chosen algorithm expects, and the package supplies the iteration logic, termination handling, and (where applicable) adaptive step-size or line-search machinery.
This design keeps the package lightweight and composable: new algorithms can be added by implementing a small, self-contained iterator interface, and existing algorithms can be reused as subproblem solvers inside other packages.
[StructuredOptimization.jl](https://github.com/JuliaFirstOrder/StructuredOptimization.jl) provides a high-level modeling language and a user-friendly interface to invoke `ProximalAlgorithms.jl`.

## State of the field

`ProximalAlgorithms.jl` has been under development since October 2017, positioning it as one of the earlier general-purpose toolboxes offering a broad, unified interface to first-order splitting methods.
`ProximalAlgorithms.jl` gives users of the Julia optimization ecosystems direct access to a wide range of well-established, first-order splitting algorithms through a single, consistent, allocation-conscious interface, backed by [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)'s extensive library of proximable functions.

This complements, rather than duplicates, other optimization software in the Julia ecosystem.
[RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) [@gollier2026] implements model-based trust-region and quadratic-regularization methods for a similar class of nonsmooth problems.
Such methods typically require fewer evaluations of the smooth term and its gradient than the line-search-based methods in `ProximalAlgorithms.jl`, at the cost of more proximal-operator evaluations.
[FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl) is a toolbox for convex optimization using conditional gradient algorihms [@besancon2022], which rely on linear minimization oracles instead of proximal mappings.
[Manopt.jl](https://manoptjl.org/) [@bergmann2022] provides algorithms for optimization problems given on a Riemannian manifold and includes some splitting-based methods.

There are also related Python packages that implement proximal methods, often restricted to convex problems or designed for specific applications, such as 
[SCICO](https://github.com/lanl/scico) [@scico2022] and
[PyProximal](https://github.com/PyLops/pyproximal) [@pyproximal2024].

## Research impact

Since its initial release, `ProximalAlgorithms.jl` has grown both its user base and contributor community, while supporting research in control and signal processing as well as for advancing optimization methods.
[@antonello2019] address joint acoustic source localization and dereverberation via sparse regularization.
In control, the `PANOC` solver has been used for embedded nonlinear model predictive control [@sathya2018], distributed motion planning at road intersections [@katriniok2019], and Gauss-Newton-accelerated nonlinear optimal control [@pas2023].
Within optimization, [@demarchi2023] builds on the `PANOCplus` solver to implement augmented Lagrangian methods for constrained nonsmooth problems.
The package has also seen adoption beyond its original developer community:
[@liang2025] invokes `FISTA` as a subroutine for their numerical scheme,
[@waldmann2026] use it to implement their proximal multi-objective method in genomic prediction,
[@adeoye2026] benchmark their smoothing approach against several proximal-gradient algorithms in `ProximalAlgorithms.jl`.

# Example

The following condensed example minimizes the two-dimensional Rosenbrock function added to a $\ell_1$-norm regularizer, using the accelerated (fast) forward-backward splitting method:

```julia
using ProximalOperators, ProximalAlgorithms

struct Rosenbrock{T}
  a::T
end
(f::Rosenbrock)(x) = (f.a-x[1])^2 + 100*(x[2]-x[1]^2)^2
function ProximalAlgorithms.value_and_gradient(f::Rosenbrock, x)
  return f(x), [2*(x[1]-f.a) + 400*(x[1]^2-x[2])*x[1], 200*(x[2]-x[1]^2)]
end

f = Rosenbrock(1.0)
g = NormL1(2.0)

solver = ProximalAlgorithms.FastForwardBackward(tol = 1e-5, verbose = true)
x, iters = solver(x0 = ones(2), f = f, g = g)
```

Here `f` wraps the smooth Rosenbrock term and `g` is the scaled $\ell_1$-norm, whose proximal mapping is the coordinate-wise soft-thresholding operator.
The `FastForwardBackward` algorithm is instantiated with a termination tolerance and verbosity option, and then called with the initial point and the two objective terms to produce the solution and the iteration count. 
Instead of providing the gradient of $f$, one can wrap the smooth cost for automatic differentiation, as with `f_auto`:

```julia
using Zygote
using DifferentiationInterface: AutoZygote

f_auto = ProximalAlgorithms.AutoDifferentiable(x -> f(x), AutoZygote())
x, iters = solver(x0 = ones(2), f = f_auto, g = g)
```

## Numerical illustration

We compare proximal-gradient solvers `ForwardBackward`, `FastForwardBackward`, `ZeroFPR`, `PANOC`, and `PANOCplus` on a regularized line spectra estimation problem, which seeks to recover the frequencies and amplitudes of a signal that consists of a mixture of sinusoids.
The least-squares cost function $f$ is complemented with different regularizers $g$ from `ProximalOperators.jl`, either convex (`NormLinf`) or nonconvex (`IndBallL0`).
Experiments were performed with Ubuntu 64-bit on an Intel Core i7 (8-core) machine, using Julia 1.13.2.

The tables report, for each regularizer, the convergence status of each solver, the runtime, the number of function, gradient, and proximal evaluations, and the final objective value.
For `ZeroFPR`, `PANOC`, and `PANOCplus`, we use the limited-memory BFGS Hessian approximation (default with memory 5).

\begin{center}
  `NormLinf`
  
  \input{examples/table_NormLinf.tex}

  `IndBallL0`
  
  \input{examples/table_IndBallL0.tex}
\end{center}

Although the final objective values differ, in some cases, due to the nonconvexity of the problem, all solvers successfully returned an approximate first-order stationary point, within the specified tolerance of $10^{-6}$.

# Acknowledgements

We acknowledge contributions from Ellis Brown, Guillaume Dalle, William Kong, Paulo J. S. Silva, and Tamás Hakkel, and support from Panagiotis Patrinos and Andreas Themelis during the genesis of this project.

# AI usage disclosure

No generative AI tools were used in the development of this software or the preparation of supporting materials.
Claude Sonnet 5 was adopted to improve readability and proofread this text.
All AI-generated suggestions were critically reviewed, modified where necessary, and fully validated by the authors, who retain complete responsibility.

# References
