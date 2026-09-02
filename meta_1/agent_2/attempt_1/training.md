# Attempt 1 training

All correction constants were fit only against the complete I80 training split. The complete validation split was evaluated only after each fit. The held-out test split was never requested.

| Baseline | Expression | Constants | E_rho | E_v | E_data | Fit evals | Optimizer | Runtime (s) | Peak RSS (MB) | Feasible |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| greenshields | `exp(a*rho*(1-rho/0.55995123))` | a=0.05870392846 | 7.760090 | 6.639808 | 7.199949 | 36 | success: Optimization terminated successfully. | 7.175 | 767.06 | yes |
| idm | `exp(a*rho*(1-rho/0.694751529445))` | a=-1.069446152 | 7.201644 | 6.114647 | 6.658146 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 30.546 | 1229.26 | yes |
| weidmann | `exp(a*rho*(1-rho/0.80612097))` | a=0.1365440521 | 8.205365 | 6.206141 | 7.205753 | 52 | success: Optimization terminated successfully. | 8.866 | 1407.50 | yes |
| triangular | `exp(a*rho*(1-rho/0.671299943071))` | a=-0.5717219093 | 7.716850 | 9.789294 | 8.753073 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 10.632 | 1586.09 | yes |
| del_castillo | `exp(a*rho*(1-rho/0.61532169))` | a=-0.05711308322 | 7.384115 | 6.707157 | 7.045636 | 43 | success: Optimization terminated successfully. | 12.058 | 1823.37 | yes |

Optimizer: SciPy Powell, two deterministic restarts (the zero/identity start plus one seeded uniform start), 30 function evaluations per restart, and coefficient bounds `[-6, 6]`. The logged evaluation count is summed across both restarts. Runtime includes fitting and final train/validation evaluations; RSS is the process high-water mark reported by the evaluator.
