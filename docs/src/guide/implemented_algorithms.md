```@meta
CurrentModule = ProximalAlgorithms
```
# [Problem types and algorithms](@id problems_algorithms)

!!! warning

    This page is under construction, and may be incomplete.

Depending on the structure a problem can be reduced to, different types of algorithms will apply.
The major distinctions are in the number of objective terms, whether any of them is differentiable,
whether they are composed with some linear mapping (which in general complicates evaluating the proximal mapping).
Based on this we can split problems, and algorithms that apply to them, in three categories:
- [Two-terms: ``f + g``](@ref two_terms_splitting)
- [Three-terms: ``f + g + h``](@ref three_terms_splitting)
- [Primal-dual: ``f + g + h \circ L``](@ref primal_dual_splitting)

In what follows, the list of available algorithms is given, with links to the documentation for their constructors
and their underlying [iterator type](@ref iterator_interface).

## [Two-terms: ``f + g``](@id two_terms_splitting)

This is the most popular model, by far the most thoroughly studied, and an abundance of algorithms exist to solve problems in this form.

Algorithm | Assumptions | Oracle | Implementation | References
----------|-------------|--------|----------------|-----------
Proximal gradient | ``f`` smooth | ``\nabla f``, ``\operatorname{prox}_{\gamma g}`` | [`ForwardBackward`](@ref) | [Lions1979](@cite)
Douglas-Rachford | | ``\operatorname{prox}_{\gamma f}``, ``\operatorname{prox}_{\gamma g}`` | [`DouglasRachford`](@ref) | [Eckstein1992](@cite)
Fast proximal gradient | ``f`` convex, smooth, ``g`` convex | ``\nabla f``, ``\operatorname{prox}_{\gamma g}`` | [`FastForwardBackward`](@ref) | [Tseng2008](@cite), [Beck2009](@cite)
PANOC | ``f`` smooth | ``\nabla f``, ``\operatorname{prox}_{\gamma g}`` | [`PANOC`](@ref) | [Stella2017](@cite)
ZeroFPR | ``f`` smooth | ``\nabla f``, ``\operatorname{prox}_{\gamma g}`` | [`ZeroFPR`](@ref) | [Themelis2018](@cite)
Douglas-Rachford line-search | ``f`` smooth | ``\operatorname{prox}_{\gamma f}``, ``\operatorname{prox}_{\gamma g}`` | [`DRLS`](@ref) | [Themelis2020](@cite)
PANOC+ | ``f`` locally smooth | ``\nabla f``, ``\operatorname{prox}_{\gamma g}`` | [`PANOCplus`](@ref) | [DeMarchi2021](@cite)

```@docs
ProximalAlgorithms.ForwardBackward
ProximalAlgorithms.ForwardBackwardIteration
ProximalAlgorithms.DouglasRachford
ProximalAlgorithms.DouglasRachfordIteration
ProximalAlgorithms.FastForwardBackward
ProximalAlgorithms.FastForwardBackwardIteration
ProximalAlgorithms.PANOC
ProximalAlgorithms.PANOCIteration
ProximalAlgorithms.ZeroFPR
ProximalAlgorithms.ZeroFPRIteration
ProximalAlgorithms.DRLS
ProximalAlgorithms.DRLSIteration
ProximalAlgorithms.PANOCplus
ProximalAlgorithms.PANOCplusIteration
```

## [Three-terms: ``f + g + h``](@id three_terms_splitting)

When more than one non-differentiable term is there in the objective, algorithms from the [previous section](@ref two_terms_splitting)
do not *in general* apply out of the box, since ``\operatorname{prox}_{\gamma (g + h)}`` does not have a closed form unless in particular cases.
Therefore, ad-hoc iteration schemes have been studied.

Algorithm | Assumptions | Oracle | Implementation | References
----------|-------------|--------|----------------|-----------
Davis-Yin | ``f`` convex and smooth, ``g, h`` convex | ``\nabla f``, ``\operatorname{prox}_{\gamma g}``, ``\operatorname{prox}_{\gamma h}`` | [`DavisYin`](@ref) | [Davis2017](@cite) 

```@docs
ProximalAlgorithms.DavisYin
ProximalAlgorithms.DavisYinIteration
```

## [Primal-dual: ``f + g + h \circ L``](@id primal_dual_splitting)

When a function ``h`` is composed with a linear operator ``L``, the proximal operator of ``h \circ L`` does not have a closed form in general.
For this reason, specific algorithms by the name of "primal-dual" splitting schemes are often applied to this model.

Algorithm | Assumptions | Oracle | Implementation | References
----------|-------------|--------|----------------|-----------
Chambolle-Pock | ``f\equiv 0``, ``g, h`` convex, ``L`` linear operator | ``\operatorname{prox}_{\gamma g}``, ``\operatorname{prox}_{\gamma h}``, ``L``, ``L^*`` | [`ChambollePock`](@ref) | [Chambolle2011](@cite)
Vu-Condat | ``f`` convex and smooth, ``g, h`` convex, ``L`` linear operator | ``\nabla f``, ``\operatorname{prox}_{\gamma g}``, ``\operatorname{prox}_{\gamma h}``, ``L``, ``L^*`` | [`VuCodat`](@ref) | [Vu2013](@cite), [Condat2013](@cite)
AFBA      | ``f`` convex and smooth, ``g, h`` convex, ``L`` linear operator | ``\nabla f``, ``\operatorname{prox}_{\gamma g}``, ``\operatorname{prox}_{\gamma h}``, ``L``, ``L^*`` | [`AFBA`](@ref) | [Latafat2017](@cite)

```@docs
ProximalAlgorithms.ChambollePock
ProximalAlgorithms.ChambollePockIteration
ProximalAlgorithms.VuCondat
ProximalAlgorithms.VuCondatIteration
ProximalAlgorithms.AFBA
ProximalAlgorithms.AFBAIteration
```
