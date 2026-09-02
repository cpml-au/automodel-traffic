# Attempt 3 evaluation

The fitted constants were frozen and evaluated once on the complete validation
split (times 64--107), with no test evaluation. Fitness adds the 0.20 complexity
penalty (`0.01 * 20` nodes) to validation data error.

| Baseline | Validation density error | Validation velocity error | Validation data error | Validation fitness | Peak RSS (MB) | Feasible |
|---|---:|---:|---:|---:|---:|:---:|
| Greenshields | 10.473258 | 7.642451 | 9.057855 | 9.257855 | 891.355 | yes |
| IDM | 7.061203 | 6.348059 | 6.704631 | 6.904631 | 1363.211 | yes |
| Weidmann | 6.543904 | 6.305277 | 6.424590 | 6.624590 | 1584.633 | yes |
| Triangular | 6.515769 | 9.196405 | 7.856087 | 8.056087 | 1801.879 | yes |
| Del Castillo | 6.051924 | 7.966357 | 7.009141 | 7.209141 | 2047.910 | yes |

The cubic family is this lineage's best IDM structure by validation data error,
but still does not beat identity after its substantial complexity penalty. The
extra terms collapsed to zero for Greenshields, Weidmann, and Triangular; Del
Castillo repeated the quadratic solution. These degeneracies favor pruning to
attempt 1 whenever the validation errors coincide. Peak RSS is cumulative.
