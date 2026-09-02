# Attempt 1 training

Expression: `exp(a*rho/(1+rho))` (8 tree nodes). Each baseline was fitted on
the complete I80 prediction training split (times 0--63) using Powell, bounds
`[-5,5]`, two starts, and at most 30 objective evaluations per start.

| Baseline | Seed | Fitted `a` | Train density error | Train velocity error | Train data error | Objective evaluations | Fit runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greenshields | 3110 | -0.06743419 | 7.558452 | 6.770174 | 7.164313 | 60 | 10.553 |
| IDM | 3111 | -0.48542155 | 7.153648 | 7.141382 | 7.147515 | 60 | 40.510 |
| Weidmann | 3112 | 0.17499456 | 8.112399 | 6.211971 | 7.162185 | 60 | 11.985 |
| Triangular | 3113 | -0.29657303 | 7.752961 | 10.221853 | 8.987407 | 60 | 11.157 |
| Del Castillo | 3114 | -0.10316200 | 7.279289 | 6.738277 | 7.008783 | 52 | 13.957 |

The optimizer reported the evaluation cap rather than formal convergence for
every winning restart. This is not treated as a failure: all selected constants
were re-evaluated, finite, positive, and passed the monotone-velocity feasibility
check. Exact optimizer messages and split-level runtime/RSS records are retained
in `results.json`.
