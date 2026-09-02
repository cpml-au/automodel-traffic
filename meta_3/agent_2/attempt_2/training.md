# Attempt 2 training

Constants were fitted on all I80 training times 0--63. Validation was evaluated only after fitting; the test interval remained untouched.

| FD | Expression | Parameters | E_rho | E_v | E_data | Evals | Optimizer | Runtime (s) | Peak RSS (MB) | Feasible |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Greenshields | `exp(a*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-14.60026637 | 7.428666 | 6.792892 | 7.110779 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 20.194 | 3919.90 | yes |
| IDM | `exp(a*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-37.11162586 | 7.052133 | 8.499180 | 7.775656 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 249.610 | 3955.52 | yes |
| Weidmann | `exp(a*St_oneD1(SquareD1(St_oneP0(rho))))` | a=0.07123095298 | 8.326215 | 6.137922 | 7.232068 | 80 | success: Optimization terminated successfully. | 17.169 | 3955.52 | yes |
| Triangular | `exp(a*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-2.086540996 | 7.850097 | 10.595211 | 9.222654 | 75 | success: Optimization terminated successfully. | 17.603 | 3955.52 | yes |
| Del Castillo | `exp(a*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-16.94716585 | 7.172242 | 6.755861 | 6.964051 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 27.626 | 3978.74 | yes |

Optimizer: SciPy Powell through the shared `fit_candidate`, two deterministic restarts (zero plus seeded uniform), 45 evaluations per restart. Bounds and seeds are recorded in `results.json`.

Feasibility: shared `is_nonlocal_feasible` homogeneous-state check, followed by finite full train and validation simulations.
