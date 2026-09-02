# Attempt 3 training

Expression: `exp((a*rho+b*rho^2+c*rho^3)/(1+rho))` (20 tree nodes). Each
baseline was fitted on the complete I80 prediction training split (times 0--63)
using Powell, bounds `[-5,5]` for all constants, two starts, and at most 30
objective evaluations per start.

| Baseline | Seed | Fitted `(a,b,c)` | Train density error | Train velocity error | Train data error | Evaluations | Runtime (s) |
|---|---:|---|---:|---:|---:|---:|---:|
| Greenshields | 3130 | (-0.06743419, 0, 0) | 7.558452 | 6.770174 | 7.164313 | 60 | 12.780 |
| IDM | 3131 | (-2.08715098, 4.31119890, -1.89173055) | 7.387383 | 5.878190 | 6.632787 | 60 | 42.672 |
| Weidmann | 3132 | (0.17499456, 0, 0) | 8.112399 | 6.211971 | 7.162185 | 60 | 12.441 |
| Triangular | 3133 | (-0.29657303, 0, 0) | 7.752961 | 10.221853 | 8.987407 | 60 | 12.453 |
| Del Castillo | 3134 | (-0.10323082, -0.10002659, 0) | 7.221415 | 6.780104 | 7.000760 | 60 | 16.224 |

All winning restarts reached the function-evaluation cap and thus reported
`optimizer_success=false`; this is a bounded-budget status rather than a
numerical failure. Final simulations were finite and all candidates passed the
physical feasibility check. Exact optimizer/runtime/RSS records are in
`results.json`.
