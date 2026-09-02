# Attempt 2 training

Expression: `exp(c0 + a*rho)` (6 tree nodes).

All values use the full I80 prediction training split (times 0-63).
Fits used two deterministic Powell starts and at most 45 function
evaluations per start. Parameter bounds: `[[-1.0, 1.0], [-5.0, 5.0]]`.

| Baseline | Constants | E_rho | E_v | E_data | Feasible | Evaluations | Optimizer status | Fit runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---|---:|---:|
| Greenshields | c0=0.02312297116, a=-0.08831739726 | 7.453785 | 6.653316 | 7.053550 | yes | 90 | False: Maximum number of function evaluations has been exceeded. | 13.838 | 2278.5 |
| IDM | c0=-0.1678685824, a=-0.09076265524 | 7.150695 | 6.096747 | 6.623721 | yes | 90 | False: Maximum number of function evaluations has been exceeded. | 44.254 | 2815.1 |
| Weidmann | c0=-0.07011210945, a=0.2297073667 | 7.921928 | 5.862733 | 6.892331 | yes | 90 | False: Maximum number of function evaluations has been exceeded. | 15.856 | 3092.5 |
| Triangular | c0=-0.1299587479, a=0.1091785095 | 7.873022 | 9.480879 | 8.676950 | yes | 90 | False: Maximum number of function evaluations has been exceeded. | 15.953 | 3346.7 |
| Del Castillo | c0=0.005495202656, a=-0.09005362272 | 7.230448 | 6.733481 | 6.981965 | yes | 90 | False: Maximum number of function evaluations has been exceeded. | 23.587 | 3712.4 |

Attempt wall time: 113.490 seconds.
