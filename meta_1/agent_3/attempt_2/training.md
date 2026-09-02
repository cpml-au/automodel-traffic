# Attempt 2 training

Expression: `exp((a*rho+b*rho^2)/(1+rho))` (13 tree nodes). Each baseline was
fitted on the complete I80 prediction training split (times 0--63) using Powell,
bounds `[-5,5]` for both constants, two starts, and at most 30 objective
evaluations per start.

| Baseline | Seed | Fitted `(a,b)` | Train density error | Train velocity error | Train data error | Evaluations | Runtime (s) |
|---|---:|---|---:|---:|---:|---:|---:|
| Greenshields | 3120 | (-0.06743419, 0) | 7.558452 | 6.770174 | 7.164313 | 60 | 12.841 |
| IDM | 3121 | (-0.48574538, 0.25587466) | 7.030916 | 7.103953 | 7.067435 | 60 | 43.678 |
| Weidmann | 3122 | (0.14989495, 0.08306538) | 8.094043 | 6.181992 | 7.138017 | 60 | 12.901 |
| Triangular | 3123 | (-0.29657303, 0) | 7.752961 | 10.221853 | 8.987407 | 60 | 12.743 |
| Del Castillo | 3124 | (-0.10323082, -0.10002659) | 7.221415 | 6.780104 | 7.000760 | 60 | 16.442 |

All winning restarts reached the evaluation cap rather than formal Powell
convergence. All final candidates nevertheless had finite train evaluations and
passed positivity/non-negative monotone-velocity feasibility. Full records,
including optimizer messages and process RSS, are in `results.json`.
