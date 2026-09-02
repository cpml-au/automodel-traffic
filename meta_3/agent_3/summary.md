# Meta 3 agent 3 summary

This lineage held every selected meta-2 incumbent coefficient fixed and fitted
only incremental DEC-convolution coefficients. All 15 train/validation fits
were harvested; the held-out test interval was never evaluated. Fourteen final
fits passed the nonlocal homogeneous-state and finite-simulation checks. The
attempt-2 Weidmann optimizer ended at the positive bound with a non-monotone
homogeneous corrected velocity and is rejected as infeasible.

## Validation results

Fitness includes the complete incumbent-plus-increment expression tree penalty.

| Baseline | Incumbent fitness | Attempt 1 fitness | Attempt 2 fitness | Attempt 3 fitness | Selection after this lineage |
|---|---:|---:|---:|---:|---|
| Greenshields | 7.627391 | 8.263538 | **6.953732** | 7.289650 | attempt 2 |
| IDM | 5.955012 | 7.755396 | **5.417018** | 6.099699 | attempt 2 |
| Weidmann | **6.322293** | 7.789705 | rejected (infeasible) | 7.518819 | incumbent |
| Triangular | **6.457489** | 7.772596 | 7.801809 | 7.453249 | incumbent |
| Del Castillo | 6.603956 | 7.067829 | 5.895957 | **5.730221** | attempt 3 |

## Improvements over incumbents

| Baseline | Best accepted incremental expression | New coefficients | Validation E_data | Validation fitness | Fitness change |
|---|---|---|---:|---:|---:|
| Greenshields | `g_inc*exp(a*(conv_3-3*conv_1))` | `a=-293.4581091` | 6.703732 | 6.953732 | **-0.673659 (-8.83%)** |
| IDM | `exp(a*(conv_3-3*conv_1))` | `a=-46.9914360` | 5.277018 | 5.417018 | **-0.537994 (-9.03%)** |
| Weidmann | no incremental model accepted | -- | 6.222293 | 6.322293 | 0.000000 |
| Triangular | no incremental model accepted | -- | 6.447489 | 6.457489 | 0.000000 |
| Del Castillo | `g_inc*exp(a*conv_3+b*(conv_3-3*conv_1))` | `a=-2.1218116, b=-256.8692295` | 5.520221 | 5.730221 | **-0.873735 (-13.23%)** |

The level-cancelling convolution contrast is the useful feature in this
lineage. It improves Greenshields, IDM, and Del Castillo substantially, but not
Weidmann or Triangular. The fitted contrast coefficients for Greenshields and
Del Castillo are large and near the search bounds, so the root reviewer should
re-evaluate these candidates and compare them with other meta-3 lineages before
promotion.

All simulations used the row-wise sum of the absolute full flux Jacobian as the
Rusanov speed bound, including off-diagonal convolution coupling. Results and
diagnostics are in each `attempt_i/results.json`; `test_evaluated=false` is
recorded at the protocol, attempt, and per-baseline levels.
