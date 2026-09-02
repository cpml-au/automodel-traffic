# Attempt 2 training

Expression: `exp(a*rho + b*rho^2)` (10 tree nodes).

All values below use the full I80 prediction training split. Fits used
two Powell starts, 30 evaluations per start, and parameter bounds `[-4, 4]`.

| Baseline | Constants | rho error | velocity error | data error | Feasible | Evaluations | Optimizer status | Fit runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---|---:|---:|
| Greenshields | a=-0.05944921716, b=0 | 7.507605 | 6.794767 | 7.151186 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 9.708 | 2025.4 |
| IDM | a=-0.9796330922, b=1.451517799 | 7.166966 | 6.142354 | 6.654660 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 30.118 | 2378.2 |
| Weidmann | a=0.1633383861, b=0.3654981168 | 7.768281 | 6.277887 | 7.023084 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 10.647 | 2548.3 |
| Triangular | a=-0.1937875339, b=0 | 7.765419 | 10.349940 | 9.057680 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 9.698 | 2732.1 |
| Del Castillo | a=0.1435023183, b=-0.4919838371 | 7.128882 | 6.750041 | 6.939462 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 17.127 | 2971.9 |

Attempt wall time: 77.301 seconds.
