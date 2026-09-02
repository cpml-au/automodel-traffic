# Attempt 2 training

Constants were fit only on the complete I80 training split. Validation was scored only after fitting, and the held-out test split was never evaluated.

| Baseline | Exact expression | Constants | E_rho | E_v | E_data | Fit evals | Optimizer | Fit runtime (s) | Peak RSS (MB) | Feasible |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| greenshields | `exp(a*(rho-0.279975615) + b*(rho-0.279975615)^2)` | a=2.240148284, b=1.329355972 | 10.873454 | 5.356174 | 8.114814 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 14.596 | 2288.91 | yes |
| idm | `exp(a*(rho-0.347375764723) + b*(rho-0.347375764723)^2)` | a=3.146724644, b=3.765862361 | 10.490432 | 5.255686 | 7.873059 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 45.978 | 2792.16 | yes |
| weidmann | `exp(a*(rho-0.403060485) + b*(rho-0.403060485)^2)` | a=0.4912620529, b=0.09017511772 | 7.575188 | 5.733812 | 6.654500 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 15.067 | 3049.41 | yes |
| triangular | `exp(a*(rho-0.335649971536) + b*(rho-0.335649971536)^2)` | a=0.4085908821, b=3.54168635 | 7.741910 | 9.950283 | 8.846097 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 16.676 | 3312.15 | yes |
| del_castillo | `exp(a*(rho-0.307660845) + b*(rho-0.307660845)^2)` | a=2.288143911, b=3.222717343 | 11.189446 | 5.523811 | 8.356628 | 72 | success: Optimization terminated successfully. | 19.310 | 3619.21 | yes |

Optimizer: SciPy Powell with two deterministic restarts (zero/identity and one seeded uniform start), 45 function evaluations per restart, and bounds `[-5, 5]` for every coefficient. The fit-evaluation count is summed across restarts. Runtime includes fitting and final train/validation evaluations; RSS is the evaluator's process high-water mark.
