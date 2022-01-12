# [Problem types and algorithms](@id problems_algorithms)

!!! warning

    This page is under construction, and may be incomplete.

Depending on the structure a problem can be reduced to, different types of algorithms will apply.
The major distinctions are in the number of objective terms, whether any of them is differentiable,
whether they are composed with some linear mapping (which in general complicates evaluating the proximal mapping).

## [Two-terms: ``f + g``](@id two_terms_splitting)

This is the most popular model, by far the most thoroughly studied, and an abundance of algorithms exist to solve problems in this form.

Algorithm | Assumptions | Oracle | Implementation | References
----------|-------------|--------|----------------|-----------
Forward-backward | ``f`` smooth | ``\nabla f``, ``\operatorname{prox}_{\gamma g}`` | [`ForwardBackwardIteration`](@ref ProximalAlgorithms.ForwardBackwardIteration) | [Lions1979](@cite)
Douglas-Rachford | | ``\operatorname{prox}_{\gamma f}``, ``\operatorname{prox}_{\gamma g}`` | [`DouglasRachfordIteration`](@ref ProximalAlgorithms.DouglasRachfordIteration) | [Eckstein1992](@cite)
Fast forward-backward | ``f`` convex, smooth, ``g`` convex | ``\nabla f``, ``\operatorname{prox}_{\gamma g}`` | [`FastForwardBackwardIteration`](@ref ProximalAlgorithms.FastForwardBackwardIteration) | [Tseng2008](@cite), [Beck2009](@cite)
PANOC | ``f`` smooth | ``\nabla f``, ``\operatorname{prox}_{\gamma g}`` | [`PANOCIteration`](@ref ProximalAlgorithms.PANOCIteration) | [Stella2017](@cite)
ZeroFPR | ``f`` smooth | ``\nabla f``, ``\operatorname{prox}_{\gamma g}`` | [`ZeroFPRIteration`](@ref ProximalAlgorithms.ZeroFPRIteration) | [Themelis2018](@cite)
Douglas-Rachford line-search | ``f`` smooth | ``\operatorname{prox}_{\gamma f}``, ``\operatorname{prox}_{\gamma g}`` | [`DRLSIteration`](@ref ProximalAlgorithms.DRLSIteration) | [Themelis2020](@cite)
PANOC+ | ``f`` locally smooth | ``\nabla f``, ``\operatorname{prox}_{\gamma g}`` | [`PANOCplusIteration`](@ref ProximalAlgorithms.PANOCplusIteration) | [DeMarchi2021](@cite)

```@docs
ProximalAlgorithms.ForwardBackwardIteration
ProximalAlgorithms.DouglasRachfordIteration
ProximalAlgorithms.FastForwardBackwardIteration
ProximalAlgorithms.PANOCIteration
ProximalAlgorithms.ZeroFPRIteration
ProximalAlgorithms.DRLSIteration
ProximalAlgorithms.PANOCplusIteration
```

## [Three-terms: ``f + g + h``](@id three_terms_splitting)

When more than one non-differentiable term is there in the objective, algorithms from the [previous section](@ref two_terms_splitting)
do not *in general* apply out of the box, since ``\operatorname{prox}_{\gamma (f + g)}`` does not have a closed form unless in particular cases.
Therefore, ad-hoc iteration schemese have been studied.

Algorithm | Assumptions | Oracle | Implementation | References
----------|-------------|--------|----------------|-----------
Davis-Yin | ``f, g`` convex, ``h`` convex and smooth | ``\operatorname{prox}_{\gamma f}``, ``\operatorname{prox}_{\gamma g}``, ``\nabla h`` | [`DavisYinIteration`](@ref ProximalAlgorithms.DavisYinIteration) | [Davis2017](@cite) 

```@docs
ProximalAlgorithms.DavisYinIteration
```

## [Primal-dual: ``f + g + h \circ L``](@id primal_dual_splitting)

When a function ``h`` is composed with a linear operator ``L``, the proximal operator of ``h \circ L`` does not have a closed form in general.
For this reason, specific algorithms by the name of "primal-dual" splitting schemes are often applied to this model.

Algorithm | Assumptions | Oracle | Implementation | References
----------|-------------|--------|----------------|-----------
Vu-Condat | ``f`` convex and smooth, ``g, h`` convex, ``L`` linear operator | ``\nabla f``, ``\operatorname{prox}_{\gamma g}``, ``\operatorname{prox}_{\gamma h}``, ``L``, ``L^*`` | [`VuCodatIteration`](@ref ProximalAlgorithms.VuCondatIteration) | [Vu2013](@cite), [Condat2013](@cite)
AFBA      | ``f`` convex and smooth, ``g, h`` convex, ``L`` linear operator | ``\nabla f``, ``\operatorname{prox}_{\gamma g}``, ``\operatorname{prox}_{\gamma h}``, ``L``, ``L^*`` | [`AFBAIteration`](@ref ProximalAlgorithms.AFBAIteration) | [Latafat2017](@cite)

```@docs
ProximalAlgorithms.AFBAIteration
ProximalAlgorithms.VuCondatIteration
```
