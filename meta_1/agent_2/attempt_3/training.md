# Attempt 3 training

All correction constants were fit only against the complete I80 training split. The complete validation split was evaluated only after each fit. The held-out test split was never requested.

| Baseline | Expression | Constants | E_rho | E_v | E_data | Fit evals | Optimizer | Runtime (s) | Peak RSS (MB) | Feasible |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| greenshields | `exp((a*rho+b*rho^2+c*rho^3)*(1-rho/0.55995123))` | a=0.05761356535, b=-0.4946993601, c=0 | 7.508943 | 6.723948 | 7.116446 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 11.002 | 841.13 | yes |
| idm | `exp((a*rho+b*rho^2+c*rho^3)*(1-rho/0.694751529445))` | a=-1.436972907, b=-0.6452147119, c=4.43558174 | 7.434977 | 5.772016 | 6.603496 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 30.672 | 1319.74 | yes |
| weidmann | `exp((a*rho+b*rho^2+c*rho^3)*(1-rho/0.80612097))` | a=0.1365440501, b=0, c=0 | 8.205365 | 6.206141 | 7.205753 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 12.748 | 1527.81 | yes |
| triangular | `exp((a*rho+b*rho^2+c*rho^3)*(1-rho/0.671299943071))` | a=-0.04847206901, b=-2.62266433, c=4.92147214 | 7.738590 | 9.715961 | 8.727276 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 12.697 | 1723.36 | yes |
| del_castillo | `exp((a*rho+b*rho^2+c*rho^3)*(1-rho/0.61532169))` | a=-0.0559121568, b=-0.4454623889, c=0 | 7.276693 | 6.746301 | 7.011497 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 16.173 | 2026.91 | yes |

Optimizer: SciPy Powell, two deterministic restarts (the zero/identity start plus one seeded uniform start), 30 function evaluations per restart, and coefficient bounds `[-6, 6]`. The logged evaluation count is summed across both restarts. Runtime includes fitting and final train/validation evaluations; RSS is the process high-water mark reported by the evaluator.
