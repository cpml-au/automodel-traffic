# Attempt 3 training

Constants were fitted on all I80 training times 0--63. Validation was evaluated only after fitting; the test interval remained untouched.

| FD | Expression | Parameters | E_rho | E_v | E_data | Evals | Optimizer | Runtime (s) | Peak RSS (MB) | Feasible |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Greenshields | `exp(a*St_oneD1(St_oneP0(rho)) + b*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-0.05958052143, b=-3.543462132 | 7.455503 | 6.838492 | 7.146998 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 25.482 | 4169.61 | yes |
| IDM | `exp(a*St_oneD1(St_oneP0(rho)) + b*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-0.3038803577, b=13.71592771 | 7.019126 | 7.513211 | 7.266169 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 266.288 | 4197.74 | yes |
| Weidmann | `exp(a*St_oneD1(St_oneP0(rho)) + b*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-0.0001305520795, b=0 | 8.327096 | 6.138668 | 7.232882 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 14.647 | 4197.74 | yes |
| Triangular | `exp(a*St_oneD1(St_oneP0(rho)) + b*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-0.190950514, b=0 | 7.766304 | 10.349214 | 9.057758 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 15.327 | 4197.74 | yes |
| Del Castillo | `exp(a*St_oneD1(St_oneP0(rho)) + b*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-0.08073113162, b=-5.426086687 | 7.191931 | 6.790542 | 6.991237 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 19.012 | 4197.74 | yes |

Optimizer: SciPy Powell through the shared `fit_candidate`, two deterministic restarts (zero plus seeded uniform), 45 evaluations per restart. Bounds and seeds are recorded in `results.json`.

Feasibility: shared `is_nonlocal_feasible` homogeneous-state check, followed by finite full train and validation simulations.
