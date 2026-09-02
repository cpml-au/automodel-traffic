# Attempt 3 training

Expression: `(1 + a*rho + b*rho^2)/(1 + c*rho)` (17 tree nodes).

All values use the full I80 prediction training split (times 0--63).
Two deterministic Powell starts used 45 evaluations per start and
parameter bounds `[-0.9, 4.0]`. Feasibility was checked before every
simulation objective call.

| Baseline | Constants | rho error | velocity error | data error | Feasible | Evaluations | Infeasible rejected | Optimizer status | Fit runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---:|---|---:|---:|
| Greenshields | a=-0.05905643164, b=-0.04344204673, c=0 | 7.454811 | 6.838233 | 7.146522 | yes | 90 | 0 | False: Maximum number of function evaluations has been exceeded. | 16.801 | 960.2 |
| IDM | a=0.6804113862, b=3.488739361, c=2.933020238 | 7.439164 | 5.860003 | 6.649583 | yes | 90 | 0 | False: Maximum number of function evaluations has been exceeded. | 43.894 | 1642.1 |
| Weidmann | a=3.508096828, b=2.270760695, c=3.989828642 | 7.758569 | 5.771022 | 6.764796 | yes | 90 | 0 | False: Maximum number of function evaluations has been exceeded. | 18.523 | 1934.6 |
| Triangular | a=0.4679734922, b=3.576157952, c=2.132359604 | 8.621257 | 8.682327 | 8.651793 | yes | 90 | 0 | False: Maximum number of function evaluations has been exceeded. | 15.475 | 2229.8 |
| Del Castillo | a=0.3448258363, b=-0.3734323616, c=0.2615828572 | 7.162548 | 6.733166 | 6.947857 | yes | 90 | 0 | False: Maximum number of function evaluations has been exceeded. | 22.797 | 2588.1 |

Attempt wall time: 117.492 seconds.
