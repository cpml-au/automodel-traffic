# Attempt 1 training

Expression: `1 + a*rho` (5 tree nodes).

All values use the full I80 prediction training split (times 0--63).
Two deterministic Powell starts used 45 evaluations per start and
parameter bounds `[-0.9, 4.0]`. Feasibility was checked before every
simulation objective call.

| Baseline | Constants | rho error | velocity error | data error | Feasible | Evaluations | Infeasible rejected | Optimizer status | Fit runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---:|---|---:|---:|
| Greenshields | a=-0.05878239279 | 7.506843 | 6.794493 | 7.150668 | yes | 90 | 0 | False: Maximum number of function evaluations has been exceeded. | 14.684 | 903.2 |
| IDM | a=-0.2733786877 | 7.132698 | 7.604856 | 7.368777 | yes | 90 | 0 | False: Maximum number of function evaluations has been exceeded. | 44.156 | 1505.6 |
| Weidmann | a=0.1637536228 | 8.043331 | 6.201799 | 7.122565 | yes | 65 | 0 | True: Optimization terminated successfully. | 10.309 | 1683.4 |
| Triangular | a=-0.1803255837 | 7.768466 | 10.363767 | 9.066116 | yes | 69 | 0 | False: Maximum number of function evaluations has been exceeded. | 11.278 | 1872.5 |
| Del Castillo | a=-0.0790741062 | 7.247148 | 6.744030 | 6.995589 | yes | 90 | 0 | False: Maximum number of function evaluations has been exceeded. | 23.062 | 2198.3 |

Attempt wall time: 103.492 seconds.
