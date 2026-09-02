# Attempt 2 training

Expression: `(1 + a*rho)/(1 + b*rho)` (11 tree nodes).

All values use the full I80 prediction training split (times 0--63).
Two deterministic Powell starts used 45 evaluations per start and
parameter bounds `[-0.9, 4.0]`. Feasibility was checked before every
simulation objective call.

| Baseline | Constants | rho error | velocity error | data error | Feasible | Evaluations | Infeasible rejected | Optimizer status | Fit runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---:|---|---:|---:|
| Greenshields | a=-0.05905643164, b=-0.00012482376 | 7.506393 | 6.794942 | 7.150668 | yes | 90 | 0 | False: Maximum number of function evaluations has been exceeded. | 16.435 | 946.5 |
| IDM | a=1.803268255, b=2.551713544 | 7.157116 | 6.793112 | 6.975114 | yes | 90 | 0 | False: Maximum number of function evaluations has been exceeded. | 47.711 | 1622.9 |
| Weidmann | a=0.159189116, b=-0.00881146449 | 8.036165 | 6.208272 | 7.122218 | yes | 82 | 0 | True: Optimization terminated successfully. | 15.358 | 1881.4 |
| Triangular | a=0.197528937, b=0.4247429861 | 7.759993 | 10.304924 | 9.032458 | yes | 90 | 0 | False: Maximum number of function evaluations has been exceeded. | 16.665 | 2157.7 |
| Del Castillo | a=-0.07953900713, b=-0.0002736367539 | 7.246859 | 6.744312 | 6.995585 | yes | 84 | 0 | False: Maximum number of function evaluations has been exceeded. | 24.417 | 2490.7 |

Attempt wall time: 120.589 seconds.
