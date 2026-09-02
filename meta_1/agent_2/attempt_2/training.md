# Attempt 2 training

All correction constants were fit only against the complete I80 training split. The complete validation split was evaluated only after each fit. The held-out test split was never requested.

| Baseline | Expression | Constants | E_rho | E_v | E_data | Fit evals | Optimizer | Runtime (s) | Peak RSS (MB) | Feasible |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| greenshields | `exp((a*rho+b*rho^2)*(1-rho/0.55995123))` | a=0.05870392846, b=-0.4948637636 | 7.509015 | 6.722799 | 7.115907 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 11.117 | 843.92 | yes |
| idm | `exp((a*rho+b*rho^2)*(1-rho/0.694751529445))` | a=-1.068733663, b=0.3117903363 | 7.170616 | 6.076310 | 6.623463 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 31.875 | 1285.40 | yes |
| weidmann | `exp((a*rho+b*rho^2)*(1-rho/0.80612097))` | a=0.2102290906, b=0.4442977224 | 7.964241 | 6.357306 | 7.160773 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 13.208 | 1489.31 | yes |
| triangular | `exp((a*rho+b*rho^2)*(1-rho/0.671299943071))` | a=-1.700703545, b=3.301559808 | 8.578436 | 8.723826 | 8.651131 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 10.886 | 1680.70 | yes |
| del_castillo | `exp((a*rho+b*rho^2)*(1-rho/0.61532169))` | a=-0.0559121568, b=-0.4454623889 | 7.276693 | 6.746301 | 7.011497 | 60 | stopped: Maximum number of function evaluations has been exceeded. | 16.571 | 1967.89 | yes |

Optimizer: SciPy Powell, two deterministic restarts (the zero/identity start plus one seeded uniform start), 30 function evaluations per restart, and coefficient bounds `[-6, 6]`. The logged evaluation count is summed across both restarts. Runtime includes fitting and final train/validation evaluations; RSS is the process high-water mark reported by the evaluator.
