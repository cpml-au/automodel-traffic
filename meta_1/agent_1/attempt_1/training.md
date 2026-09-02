# Attempt 1 training

Expression: `exp(a*rho)` (4 tree nodes).

All values below use the full I80 prediction training split. Fits used
two Powell starts, 30 evaluations per start, and parameter bounds `[-4, 4]`.

| Baseline | Constants | rho error | velocity error | data error | Feasible | Evaluations | Optimizer status | Fit runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---|---:|---:|
| Greenshields | a=-0.05943973795 | 7.507633 | 6.794740 | 7.151186 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 9.537 | 812.0 |
| IDM | a=-0.3034516108 | 7.148321 | 7.529171 | 7.338746 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 28.523 | 1256.2 |
| Weidmann | a=0.1633405395 | 8.034649 | 6.201711 | 7.118180 | yes | 56 | True: Optimization terminated successfully. | 9.279 | 1427.5 |
| Triangular | a=-0.1937875339 | 7.765419 | 10.349940 | 9.057680 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 10.019 | 1611.0 |
| Del Castillo | a=-0.08049199 | 7.248034 | 6.744310 | 6.996172 | yes | 56 | False: Maximum number of function evaluations has been exceeded. | 15.018 | 1870.3 |

Attempt wall time: 72.378 seconds.
