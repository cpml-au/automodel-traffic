# Attempt 1 training

Constants were fitted on all I80 training times 0--63. Validation was evaluated only after fitting; the test interval remained untouched.

| FD | Expression | Parameters | E_rho | E_v | E_data | Evals | Optimizer | Runtime (s) | Peak RSS (MB) | Feasible |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Greenshields | `exp(a*St_oneD1(St_oneP0(rho)))` | a=-0.0595386222 | 7.507347 | 6.795024 | 7.151186 | 90 | stopped: Maximum number of function evaluations has been exceeded. | 21.027 | 969.56 | yes |
| IDM | `exp(a*St_oneD1(St_oneP0(rho)))` | a=-0.3034752215 | 7.148363 | 7.529133 | 7.338748 | 72 | stopped: Maximum number of function evaluations has been exceeded. | 180.765 | 3183.40 | yes |
| Weidmann | `exp(a*St_oneD1(St_oneP0(rho)))` | a=-0.0001305520795 | 8.327096 | 6.138668 | 7.232882 | 63 | success: Optimization terminated successfully. | 11.846 | 3297.77 | yes |
| Triangular | `exp(a*St_oneD1(St_oneP0(rho)))` | a=-0.1919797126 | 7.766006 | 10.349418 | 9.057713 | 75 | stopped: Maximum number of function evaluations has been exceeded. | 15.505 | 3480.78 | yes |
| Del Castillo | `exp(a*St_oneD1(St_oneP0(rho)))` | a=-0.08033418289 | 7.248250 | 6.744097 | 6.996173 | 75 | stopped: Maximum number of function evaluations has been exceeded. | 19.642 | 3703.15 | yes |

Optimizer: SciPy Powell through the shared `fit_candidate`, two deterministic restarts (zero plus seeded uniform), 45 evaluations per restart. Bounds and seeds are recorded in `results.json`.

Feasibility: shared `is_nonlocal_feasible` homogeneous-state check, followed by finite full train and validation simulations.
