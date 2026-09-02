# Meta-iteration 4 agent 3 summary

This non-convolution control lineage completed all 15 planned fits: three
protected-square-root structures times all five fixed meta-3 per-FD incumbents.
Every final candidate passed `is_nonlocal_feasible`, produced finite full-train
and full-validation simulations, and left I80 test times 108--179 untouched.

The executable protected square root was
`jnp.sqrt(jnp.maximum(rho, 0))`. Its typed-GP notation is `SqrtP0(rho)`.
Optimization used two deterministic Powell restarts, 60 evaluations per
restart, bounds `[-5, 5]`, and the specified `12100 + 10*attempt + baseline
index` seeds. A reported optimizer `success=false` means Powell reached the
evaluation cap; its best feasible point and complete evaluation metrics remain
recorded.

## Best protected-square-root result versus meta 3

| Baseline | Best attempt | Added factor and fitted parameters | Nodes | Train E_data | Validation E_data | Validation fitness | Meta-3 fitness | Change |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Greenshields | 1 | `exp(a*SqrtP0(rho))`, `a=0.05848173354` | 31 | 6.285900 | 6.899943 | 7.209943 | 6.953732 | +0.256211 |
| IDM | 3 | `exp(a*SqrtP0(rho)+b*rho^2)`, `a=-0.3554653767`, `b=0.3235312489` | 26 | 6.385878 | 6.752663 | 7.012663 | 5.417018 | +1.595645 |
| Weidmann | 1 | `exp(a*SqrtP0(rho))`, `a=-0.03384904440` | 16 | 7.211188 | 6.424434 | 6.584434 | 6.322293 | +0.262141 |
| Triangular | 3 | `exp(a*SqrtP0(rho)+b*rho^2)`, `a=-0.2865864788`, `b=0.6055425796` | 13 | 8.640522 | 7.583583 | 7.713583 | 6.457489 | +1.256094 |
| Del Castillo | 1 | `exp(a*SqrtP0(rho))`, `a=-0.002141597792` | 27 | 6.364108 | 5.526136 | 5.796136 | 5.730221 | +0.065915 |

Lower fitness is better, so no protected-square-root control displaces a
meta-3 incumbent. Del Castillo is closest, but its small raw validation change
does not pay for the six newly counted tree nodes. Across several FDs, the
two-parameter forms lower training error while worsening chronological
validation, consistent with overfitting rather than a missing local nonlinear
factor.

The aggregate per-fit runtime was 1044.717 seconds. The largest recorded peak
RSS was 4017.9 MB. Complete expressions, parameters, component errors,
optimizer messages/evaluation counts, runtime/RSS, feasibility diagnostics, and
`test_evaluated=false` flags are in each attempt's `results.json` and rendered
in its `training.md` and `evaluation.md`.
