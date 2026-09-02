# Attempt 3 training

Constants were fit only on the complete I80 training split. Validation was scored only after fitting, and the held-out test split was never evaluated.

| Baseline | Exact expression | Constants | E_rho | E_v | E_data | Fit evals | Optimizer | Fit runtime (s) | Peak RSS (MB) | Feasible |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| greenshields | `exp(a*(rho-0.279975615) + b*(rho-0.279975615)^2 + c*(rho-0.279975615)^3)` | a=-0.3771283145, b=1.325338078, c=0 | 7.204640 | 6.400663 | 6.802651 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 16.076 | 3862.28 | yes |
| idm | `exp(a*(rho-0.347375764723) + b*(rho-0.347375764723)^2 + c*(rho-0.347375764723)^3)` | a=3.218527463, b=4.998935975, c=-1.205137744 | 10.299992 | 5.254178 | 7.777085 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 48.118 | 4364.96 | yes |
| weidmann | `exp(a*(rho-0.403060485) + b*(rho-0.403060485)^2 + c*(rho-0.403060485)^3)` | a=0.4912620529, b=0.09017511772, c=-0.2156043328 | 7.564272 | 5.736605 | 6.650438 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 18.837 | 4634.06 | yes |
| triangular | `exp(a*(rho-0.335649971536) + b*(rho-0.335649971536)^2 + c*(rho-0.335649971536)^3)` | a=0.7689974756, b=4.993832243, c=-4.485952577 | 7.869749 | 9.332440 | 8.601095 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 18.165 | 4900.56 | yes |
| del_castillo | `exp(a*(rho-0.307660845) + b*(rho-0.307660845)^2 + c*(rho-0.307660845)^3)` | a=0.01976533509, b=2.353017936, c=-4.998935966 | 7.216575 | 6.412161 | 6.814368 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 23.829 | 5275.39 | yes |

Optimizer: SciPy Powell with two deterministic restarts (zero/identity and one seeded uniform start), 45 function evaluations per restart, and bounds `[-5, 5]` for every coefficient. The fit-evaluation count is summed across restarts. Runtime includes fitting and final train/validation evaluations; RSS is the evaluator's process high-water mark.
