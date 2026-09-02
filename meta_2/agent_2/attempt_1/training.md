# Attempt 1 training

Constants were fit only on the complete I80 training split. Validation was scored only after fitting, and the held-out test split was never evaluated.

| Baseline | Exact expression | Constants | E_rho | E_v | E_data | Fit evals | Optimizer | Fit runtime (s) | Peak RSS (MB) | Feasible |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| greenshields | `exp(a*(rho-0.279975615))` | a=-0.3764529309 | 7.116167 | 6.591080 | 6.853623 | 60 | success: Optimization terminated successfully. | 9.754 | 813.89 | yes |
| idm | `exp(a*(rho-0.347375764723))` | a=2.667470586 | 10.887645 | 5.426374 | 8.157009 | 62 | success: Optimization terminated successfully. | 29.328 | 1267.76 | yes |
| weidmann | `exp(a*(rho-0.403060485))` | a=0.4912659945 | 7.588152 | 5.735021 | 6.661587 | 73 | success: Optimization terminated successfully. | 11.772 | 1488.75 | yes |
| triangular | `exp(a*(rho-0.335649971536))` | a=0.2122322872 | 8.022305 | 9.968639 | 8.995472 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 15.389 | 1742.66 | yes |
| del_castillo | `exp(a*(rho-0.307660845))` | a=2.287139911 | 11.274239 | 5.600252 | 8.437245 | 62 | success: Optimization terminated successfully. | 16.968 | 2056.62 | yes |

Optimizer: SciPy Powell with two deterministic restarts (zero/identity and one seeded uniform start), 45 function evaluations per restart, and bounds `[-5, 5]` for every coefficient. The fit-evaluation count is summed across restarts. Runtime includes fitting and final train/validation evaluations; RSS is the evaluator's process high-water mark.
